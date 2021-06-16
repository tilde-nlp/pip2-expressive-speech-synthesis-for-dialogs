"""
Adapted from https://github.com/NVIDIA/tacotron2
"""
import torch
from torch import nn
from typing import List, Tuple


class Tacotron2Loss(nn.Module):
    def __init__(self, gate_loss_pos_weight=1.0):
        """
        :param float gate_loss_pos_weight: The weight of the positive class in gate loss. Used to adjust the stop token loss to to account for the imbalance between positive and negative classes.
        """
        super(Tacotron2Loss, self).__init__()
        self.gate_loss_pos_weight = gate_loss_pos_weight

    def forward(self, model_output: List, targets: Tuple) -> torch.Tensor:
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.gate_loss_pos_weight))(gate_out, gate_target)
        return mel_loss + gate_loss
