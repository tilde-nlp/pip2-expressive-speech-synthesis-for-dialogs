import sys
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from scipy.io.wavfile import write
import matplotlib.pylab as plt
from tacotron2_gst.hparams import create_hparams
from tacotron2_gst.layers import TacotronSTFT
from tacotron2_gst.audio_processing import griffin_lim, load_wav_to_torch, generate_mel_spectogram
from tacotron2_gst.data_utils import TextMelLoader
from tacotron2_gst.model import load_model
from tacotron2_gst.text import text_to_sequence
from waveglow.denoiser import Denoiser

from typing import Optional, Tuple


def load_tacotron_model(model_path: str, hparams, device: torch.device):
    model = load_model(hparams)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])

    model.eval()
    if device.type != 'cpu':
        model.cuda().half()
    return model


def load_waveglow_model(model_path: str, device: torch.device):
    # this is required for pickle to see glow module
    # TODO: specify correct path to waveglow folder
    sys.path.append("waveglow/")

    waveglow = torch.load(model_path, map_location=device)['model']
    # waveglow = waveglow.remove_weightnorm(waveglow)

    waveglow.eval()
    if device.type != 'cpu':
        waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()

    denoiser = Denoiser(waveglow)

    return waveglow, denoiser


def plot_data(idx, output_dir, data, figsize=(16, 4)):
    plt.clf()
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')
    plt.savefig(os.path.join(output_dir, "plots_%d.png" % idx))
    plt.close()


def compute_style_mel(audio, sampling_rate, hparams, device: torch.device):
    stft = TacotronSTFT(hparams.data.filter_length, hparams.data.hop_length, hparams.data.win_length,
                        sampling_rate=hparams.data.sampling_rate)

    style_mel = generate_mel_spectogram(audio, sampling_rate, stft, hparams.data.max_wav_value)
    style_mel = torch.unsqueeze(style_mel, 0).to(device)

    if device.type != "cpu":
        style_mel = style_mel.half()

    return style_mel


def synthesize_sequence(text,
                        model,
                        hparams,
                        device: torch.device,
                        waveglow: Optional,
                        denoiser: Optional,
                        style_wav=None,
                        speaker_id=None) -> Tuple:
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence)
    sequence = sequence.long().to(device)

    if isinstance(style_wav, dict):
        # style_wav == dictionary using the style tokens {'token1': 'value', 'token2': 'value'}
        # example {"0": 0.15, "1": 0.15, "5": -0.15}
        style_mel = style_wav
    elif style_wav is not None:
        # style_wav == audio reference
        audio, sampling_rate = load_wav_to_torch(style_wav)
        style_mel = compute_style_mel(audio, sampling_rate, hparams, device)
    else:
        style_mel = None

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(
            sequence, style_mel=style_mel, speaker_ids=speaker_id)

    # waveglow "vocoder"
    if waveglow and denoiser:
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            audio = denoiser(audio, 0.08)
            audio = 32768.0 * audio
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
    else:
        # griffin-lim as fallback
        taco_stft = TacotronSTFT(hparams.data.filter_length, hparams.data.hop_length, hparams.data.win_length,
                                 sampling_rate=hparams.data.sampling_rate)
        mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet).cpu()
        mel_decompress = mel_decompress.transpose(1, 2)
        spec_from_mel_scaling = 1000

        # NOTE:
        # Griffin lim, is always performed on cpu
        # in the case of using gpu for the rest of the synthesis, 16-bit tensors get introduced
        # next line makes sure that processor receives matching dtypes
        mel_decompress = mel_decompress.to(taco_stft.mel_basis.dtype)
        spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        griffin_iters = 60

        audio = griffin_lim(spec_from_mel[:, :, :-1], taco_stft.stft_fn, griffin_iters)

        audio = audio.squeeze()
        audio = audio.cpu().numpy()

    return audio, mel_outputs, mel_outputs_postnet, alignments


def synthesize(model,
               hparams,
               device: torch.device,
               input_file: str,
               output_dir: str,
               waveglow: Optional,
               denoiser: Optional,
               style_wav=None,
               speaker_id=None):
    sampling_rate = hparams.data.sampling_rate

    # prepare output dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o775)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [l.replace("\n", "") for l in lines]

    if hparams.use_speaker_embedding:
        if speaker_id is None:
            raise RuntimeError("Speaker id must be specified when 'use_speaker_embedding' == True")

        trainset = TextMelLoader(hparams.data.training_files, hparams)
        speaker_id = trainset.get_speaker_id(speaker_id).cuda()
        speaker_id = speaker_id[None]

    for i in tqdm(range(len(lines))):
        audio, mel_outputs, mel_outputs_postnet, alignments = \
            synthesize_sequence(lines[i],
                                model,
                                hparams,
                                device,
                                waveglow,
                                denoiser,
                                style_wav=style_wav,
                                speaker_id=speaker_id)
        plot_data(i, output_dir, data=(mel_outputs.float().data.cpu().numpy()[0],
                                       mel_outputs_postnet.float().data.cpu().numpy()[0],
                                       alignments.float().data.cpu().numpy()[0].T))

        audio_path = os.path.join(output_dir, f"synthesized_audio_{i}.wav")
        write(audio_path, sampling_rate, audio)


def main(args):
    # set device
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # loading models
    print("Loading models...")
    hparams = create_hparams(args.hparams)
    model = load_tacotron_model(args.checkpoint_path, hparams, device)
    if args.waveglow_path:
        waveglow, denoiser = load_waveglow_model(args.waveglow_path, device)
        print("Using waveglow neural vocoder")
    else:
        waveglow, denoiser = None, None
        print("No waveglow model path provided. Using Griffin-Lim as fallback.")
    print("Models loaded...")

    # synthesis
    print("Synthesizing...")
    synthesize(model, hparams, device, args.input_file, args.output_dir, waveglow, denoiser,
               style_wav=args.gst_style, speaker_id=args.speaker_id)
    print("Speech synthesis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input_file', type=str, help='Input file with text inside', required=True)
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None, required=True,
                        help="Path to tacotron checkpoint.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, required=True,
                        help="Output directory path, where plots and wavs will be put.")
    parser.add_argument("-w", "--waveglow_path", type=str, default=None, required=False,
                        help="Optional path to waveglow checkpoint. If absent, Griffin-Lim is used instead")
    parser.add_argument("-hp", "--hparams", type=str, required=False, help="comma separated name-value pairs")
    parser.add_argument("-sid", "--speaker_id", type=int, default=None)
    parser.add_argument('--gst_style', help="Wav path file for GST style reference.", default=None)
    parser.add_argument("--cuda", action='store_true', help="Add to run on gpu")
    args = parser.parse_args()

    main(args)
