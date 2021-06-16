"""
Adapted from:
- https://github.com/NVIDIA/tacotron2
- https://github.com/mozilla/TTS
"""
import random
from typing import List, Tuple

import torch
import numpy as np
import torch.utils.data

from tacotron2_gst import layers
from tacotron2_gst.text import text_to_sequence
from tacotron2_gst.utils import load_filepaths_and_text
from tacotron2_gst.audio_processing import load_wav_to_torch


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text: str, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.data.text_cleaners
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.load_mel_from_disk = hparams.data.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.data.filter_length, hparams.data.hop_length, hparams.data.win_length,
            hparams.data.n_mel_channels, hparams.data.sampling_rate, hparams.data.mel_fmin,
            hparams.data.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

        self.use_speaker_embedding = hparams.use_speaker_embedding

        if self.use_speaker_embedding:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

    def get_data_sample(self, audiopath_and_text: List) -> Tuple[torch.IntTensor, torch.Tensor, torch.Tensor]:
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        speaker_id = None
        if self.use_speaker_embedding:
            speaker_id = audiopath_and_text[2]

        text = self.get_text(text)
        mel = self.get_mel(audiopath)

        if speaker_id is not None:
            speaker_id = self.get_speaker_id(speaker_id)

        return text, mel, speaker_id

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        print("Number of speakers :", len(d))
        return d

    def get_mel(self, filename: str) -> torch.Tensor:
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)

            audio_norm = audio_norm.clone().detach()

            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text: str) -> torch.IntTensor:
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_speaker_id(self, speaker_id) -> torch.Tensor:
        return torch.LongTensor([self.speaker_ids[int(speaker_id)]])

    def __getitem__(self, index: int):
        return self.get_data_sample(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step: int):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        use_speaker_embedding = batch[0][2] is not None
        if use_speaker_embedding:
            speaker_ids = torch.LongTensor(len(batch))
        else:
            speaker_ids = None

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
            if use_speaker_embedding:
                speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, speaker_ids, gate_padded, output_lengths
