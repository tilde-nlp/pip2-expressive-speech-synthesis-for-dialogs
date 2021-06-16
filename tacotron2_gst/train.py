"""
Adapted from https://github.com/NVIDIA/tacotron2
"""
import os
import time
import argparse
import math
from numpy import finfo
from typing import Tuple

import torch
from tacotron2_gst.distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tacotron2_gst.model import Tacotron2
from tacotron2_gst.data_utils import TextMelLoader, TextMelCollate
from tacotron2_gst.loss_function import Tacotron2Loss
from tacotron2_gst.logger import Tacotron2Logger
from tacotron2_gst.hparams import create_hparams


def reduce_tensor(tensor: torch.Tensor, n_gpus: int) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus: int, rank: int, group_name: str):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.experiment.dist_backend, init_method=hparams.experiment.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams) -> Tuple:
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.data.training_files, hparams)
    valset = TextMelLoader(hparams.data.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.model.n_frames_per_step)

    if hparams.experiment.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.experiment.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory: str, log_directory: str, rank: int):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams)

    if hparams.experiment.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.experiment.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path: str, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path: str, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate: float, iteration: int, filepath: str):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def lr_exponential_decay(lr_0: float, lr_1: float, steps: int, current_step: int) -> float:
    """
    Decays learning rate exponentially from lr_0 to lr_1 in N steps, where N == epochs.
    :param float lr_0: initial learning rate
    :param float lr_1: target learning rate
    :param int steps: number of steps between lr_0 and lr_1
    :param int current_step: current step
    :return: float decayed learning rate
    """
    decay_rate = (lr_1 / lr_0) ** (1 / steps)
    return lr_0 * decay_rate ** current_step


def validate(model, criterion, valset, iteration: int, batch_size: int, n_gpus: int,
             collate_fn, logger, distributed_run: bool, rank: int):
    """Handles all the validation scoring and printing"""

    if n_gpus == 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = parse_batch(batch, device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def parse_batch(batch: Tuple, device: torch.device) -> Tuple:
    text_padded, input_lengths, mel_padded, speaker_ids, gate_padded, \
    output_lengths = batch
    text_padded = text_padded.contiguous().long().to(device)
    input_lengths = input_lengths.contiguous().long().to(device)
    max_len = torch.max(input_lengths.data).item()
    mel_padded = mel_padded.contiguous().float().to(device)
    if speaker_ids is not None:
        speaker_ids = speaker_ids.contiguous().long().to(device)
    gate_padded = gate_padded.contiguous().float().to(device)
    output_lengths = output_lengths.contiguous().long().to(device)

    return (
        (text_padded, input_lengths, mel_padded, speaker_ids, max_len, output_lengths),
        (mel_padded, gate_padded))


def train(output_directory: str, log_directory: str, checkpoint_path: str, warm_start: bool, n_gpus: int,
          rank: int, group_name: str, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.experiment.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.experiment.seed)
    torch.cuda.manual_seed(hparams.experiment.seed)

    if n_gpus == 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    model = load_model(hparams)

    model = model.to(device)

    learning_rate = hparams.experiment.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.experiment.weight_decay)

    if hparams.experiment.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.experiment.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss(gate_loss_pos_weight=hparams.experiment.gate_loss_pos_weight)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.experiment.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.experiment.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.experiment.epochs):
        print("Epoch: {}".format(epoch))

        if hparams.experiment.lr_exp_decay:
            if hparams.experiment.lr_exp_decay_offset:
                if epoch >= hparams.experiment.lr_exp_decay_offset:
                    # Decays the learning rate exponentially from lr_0 to lr_1, starting from offset
                    decay_steps = hparams.experiment.epochs - hparams.experiment.lr_exp_decay_offset
                    current_step = epoch - hparams.experiment.lr_exp_decay_offset
                    learning_rate = lr_exponential_decay(
                        hparams.experiment.learning_rate,
                        hparams.experiment.lr_exp_decay_final_learning_rate,
                        decay_steps,
                        current_step
                    )
            else:
                learning_rate = lr_exponential_decay(
                    hparams.experiment.learning_rate,
                    hparams.experiment.lr_exp_decay_final_learning_rate,
                    hparams.experiment.epochs,
                    epoch
                )

        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()

            x, y = parse_batch(batch, device)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.experiment.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.experiment.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.experiment.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.experiment.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.experiment.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.experiment.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.experiment.batch_size, n_gpus, collate_fn, logger,
                         hparams.experiment.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-hp', '--hparams', type=str,
                        required=True, help='path to model parameters')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus, 0 for CPU')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.experiment.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.experiment.cudnn_benchmark

    print("FP16 Run:", hparams.experiment.fp16_run)
    print("Dynamic Loss Scaling:", hparams.experiment.dynamic_loss_scaling)
    print("Distributed Run:", hparams.experiment.distributed_run)
    print("cuDNN Enabled:", hparams.experiment.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.experiment.cudnn_benchmark)


    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
