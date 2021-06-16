"""
Wrapper class for use in server
"""
from typing import Union

import torch
import numpy as np

from tacotron2_gst.hparams import create_hparams
from tacotron2_gst.data_utils import TextMelLoader
from tacotron2_gst.synth import load_tacotron_model, load_waveglow_model, synthesize_sequence


class TTSEngine(object):
    # noinspection PyAttributeOutsideInit
    def load_model(self,
                   taco_ckpt_path: str,
                   waveglow_ckpt_path: str,
                   config_path: str,
                   device: str) -> None:
        self.device = torch.device(device)
        self.hparams = create_hparams(config_path)
        self.taco = load_tacotron_model(taco_ckpt_path, self.hparams, self.device)
        self.waveglow, self.denoiser = load_waveglow_model(waveglow_ckpt_path, self.device)

        if self.hparams.use_speaker_embedding:
            self.multi_speaker_dataset = TextMelLoader(self.hparams.data.training_files, self.hparams)

    def speak(self, text: str) -> np.ndarray:
        audio, _, _, _ = synthesize_sequence(text,
                                             self.taco,
                                             self.hparams,
                                             self.device,
                                             self.waveglow,
                                             self.denoiser)
        return audio

    def speak_with_style(self, text: str, gst_style: Union[dict, str], speaker_id: int) -> np.ndarray:
        """
        Synthesizes text with style and/or speaker embedding
        :param text:
        :param gst_style: either dict of GST tokens and their values or path to a reference wav file
        :param speaker_id: integer id of the speaker
        :return:
        """
        if self.hparams.use_speaker_embedding:
            if speaker_id is None:
                speaker_id = 0

            speaker_id = self.multi_speaker_dataset.get_speaker_id(speaker_id).to(self.device)
            speaker_id = speaker_id[None]

        audio, _, _, _ = synthesize_sequence(text,
                                             self.taco,
                                             self.hparams,
                                             self.device,
                                             self.waveglow,
                                             self.denoiser,
                                             style_wav=gst_style,
                                             speaker_id=speaker_id)

        return audio

    def get_sample_rate(self) -> int:
        return self.hparams.data.sampling_rate
