"""
Adapted from https://github.com/NVIDIA/tacotron2
"""
import torch


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_filepaths_and_text(filename: str, split="|") -> list:
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text
