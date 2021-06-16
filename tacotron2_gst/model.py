"""
Adapted from:
- https://github.com/NVIDIA/tacotron2
- https://github.com/mozilla/TTS
"""
from math import sqrt
from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from tacotron2_gst.layers import ConvNorm, LinearNorm
from tacotron2_gst.utils import get_mask_from_lengths
from tacotron2_gst.gst_layers import GST


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters: int, attention_kernel_size: int,
                 attention_dim: int):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim: int, embedding_dim: int, attention_dim: int,
                 attention_location_n_filters: int, attention_location_kernel_size: int):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def init_states(self, inputs: torch.Tensor):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights = inputs.data.new(B, T).zero_()
        self.attention_weights_cum = inputs.data.new(B, T).zero_()

    def preprocess_inputs(self, inputs: torch.Tensor):
        return self.memory_layer(inputs)

    def get_alignment_energies(self, query: torch.Tensor, processed_memory: torch.Tensor) -> torch.Tensor:
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        # From https://github.com/mozilla/TTS
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
                                   self.attention_weights_cum.unsqueeze(1)),
                                  dim=1)

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state: torch.Tensor, memory: torch.Tensor, processed_memory: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        self.attention_weights_cum += self.attention_weights

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        self.attention_weights = attention_weights

        return attention_context


class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: List):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.data.n_mel_channels, hparams.model.postnet_embedding_dim,
                         kernel_size=hparams.model.postnet_kernel_size, stride=1,
                         padding=int((hparams.model.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.model.postnet_embedding_dim))
        )

        for i in range(1, hparams.model.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.model.postnet_embedding_dim,
                             hparams.model.postnet_embedding_dim,
                             kernel_size=hparams.model.postnet_kernel_size, stride=1,
                             padding=int((hparams.model.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.model.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.model.postnet_embedding_dim, hparams.data.n_mel_channels,
                         kernel_size=hparams.model.postnet_kernel_size, stride=1,
                         padding=int((hparams.model.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.data.n_mel_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.model.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.model.encoder_embedding_dim,
                         hparams.model.encoder_embedding_dim,
                         kernel_size=hparams.model.encoder_kernel_size, stride=1,
                         padding=int((hparams.model.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.model.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.model.encoder_embedding_dim,
                            int(hparams.model.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.data.n_mel_channels
        self.n_frames_per_step = hparams.model.n_frames_per_step
        self.encoder_embedding_dim = hparams.model.encoder_embedding_dim
        self.attention_rnn_dim = hparams.model.attention_rnn_dim
        self.decoder_rnn_dim = hparams.model.decoder_rnn_dim
        self.prenet_dim = hparams.model.prenet_dim
        self.max_decoder_steps = hparams.model.max_decoder_steps
        self.gate_threshold = hparams.model.gate_threshold
        self.p_attention_dropout = hparams.model.p_attention_dropout
        self.p_decoder_dropout = hparams.model.p_decoder_dropout

        self.gst_embedding_dim = None
        self.speaker_embedding_dim = None

        if hparams.use_gst:
            self.gst_embedding_dim = hparams.gst.gst_embedding_dim
            self.encoder_embedding_dim += self.gst_embedding_dim

        if hparams.use_speaker_embedding:
            self.speaker_embedding_dim = hparams.speaker_embedding_dim
            self.encoder_embedding_dim += self.speaker_embedding_dim

        self.prenet = Prenet(
            hparams.data.n_mel_channels * hparams.model.n_frames_per_step,
            [hparams.model.prenet_dim, hparams.model.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.model.prenet_dim + self.encoder_embedding_dim,
            hparams.model.attention_rnn_dim)

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.model.attention_dim, hparams.model.attention_location_n_filters,
            hparams.model.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.model.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.model.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.model.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.data.n_mel_channels * hparams.model.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.model.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory: torch.Tensor) -> torch.Tensor:
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)

        decoder_input = memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_()

        return decoder_input

    def initialize_decoder_states(self, memory: torch.Tensor, mask: Optional[torch.Tensor]):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = memory.data.new(B, self.attention_rnn_dim).zero_()
        self.attention_cell = memory.data.new(B, self.attention_rnn_dim).zero_()

        self.decoder_hidden = memory.data.new(B, self.decoder_rnn_dim).zero_()
        self.decoder_cell = memory.data.new(B, self.decoder_rnn_dim).zero_()

        self.attention_context = memory.data.new(B, self.encoder_embedding_dim).zero_()

        self.memory = memory
        self.processed_memory = self.attention_layer.preprocess_inputs(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs: torch.Tensor) -> torch.Tensor:
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs: List, gate_outputs: List, alignments: List) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input: torch.Tensor) -> Tuple:
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        # self.attention()
        self.attention_context = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory, self.mask)

        # end of self.attention()
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_layer.attention_weights

    def forward(self, memory: torch.Tensor, decoder_inputs: torch.Tensor, memory_lengths: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        self.attention_layer.init_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)
        self.attention_layer.init_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.experiment.mask_padding
        self.fp16_run = hparams.experiment.fp16_run
        self.n_mel_channels = hparams.data.n_mel_channels
        self.n_frames_per_step = hparams.model.n_frames_per_step
        self.num_speakers = hparams.data.num_speakers
        self.embedding = nn.Embedding(
            hparams.model.n_symbols, hparams.model.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.model.n_symbols + hparams.model.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        self.use_gst = hparams.use_gst
        self.use_speaker_embedding = hparams.use_speaker_embedding
        self.embeddings_per_sample = hparams.use_external_speaker_embedding_file

        # From https://github.com/mozilla/TTS
        # speaker embedding layer
        if self.use_speaker_embedding:
            if not self.embeddings_per_sample:
                speaker_embedding_dim = hparams.speaker_embedding_dim
                self.speaker_embedding = nn.Embedding(self.num_speakers, speaker_embedding_dim)
                self.speaker_embedding.weight.data.normal_(0, 0.3)

        # model states
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

        # global style token layers
        if self.use_gst:
            self.gst_embedding_dim = hparams.gst.gst_embedding_dim
            self.gst_num_heads = hparams.gst.gst_num_heads
            self.gst_style_tokens = hparams.gst.gst_style_tokens
            self.gst_use_speaker_embedding = hparams.gst.gst_use_speaker_embedding
            self.gst_layer = GST(num_mel=self.n_mel_channels,
                                 num_heads=self.gst_num_heads,
                                 num_style_tokens=self.gst_style_tokens,
                                 gst_embedding_dim=self.gst_embedding_dim,
                                 speaker_embedding_dim=self.gst_use_speaker_embedding)

    def parse_output(self, outputs: List, output_lengths=None) -> List:
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs: Tuple, speaker_embeddings=None) -> List:
        text_inputs, text_lengths, mels, speaker_ids, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        if self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(
                encoder_outputs,
                mels,
                speaker_embeddings if self.gst_use_speaker_embedding else None)

        if self.use_speaker_embedding:
            if not self.embeddings_per_sample:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            else:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = torch.unsqueeze(speaker_embeddings, 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs: torch.Tensor, speaker_ids=None, style_mel=None, speaker_embeddings=None) -> List:
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs,
                                               style_mel,
                                               speaker_embeddings if self.gst_use_speaker_embedding else None)
        if self.use_speaker_embedding:
            if not self.embeddings_per_sample:
                speaker_embeddings = self.speaker_embedding(speaker_ids)  # [:, None]
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    """
    GST and Multi-speaker adapted from Mozilla TTS
    """
    def compute_speaker_embedding(self, speaker_ids):
        """ Compute speaker embedding vectors """
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(
                " [!] Model has speaker embedding layer but speaker_id is not provided"
            )
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.speaker_embeddings = self.speaker_embedding(speaker_ids).unsqueeze(1)
        if hasattr(self, "speaker_project_mel") and speaker_ids is not None:
            self.speaker_embeddings_projected = self.speaker_project_mel(
                self.speaker_embeddings).squeeze(1)

    def compute_gst(self, inputs, style_input, speaker_embedding=None):
        """ Compute global style token """
        device = inputs.device
        if isinstance(style_input, dict):
            query = torch.zeros(1, 1, self.gst_embedding_dim//2).to(device)
            query = query.type(inputs.type())
            if speaker_embedding is not None:
                query = torch.cat([query, speaker_embedding.reshape(1, 1, -1)], dim=-1)

            _GST = torch.tanh(self.gst_layer.style_token_layer.style_tokens)
            gst_outputs = torch.zeros(1, 1, self.gst_embedding_dim).to(device)
            for k_token, v_amplifier in style_input.items():
                key = _GST[int(k_token)].unsqueeze(0).expand(1, -1, -1)
                key = key.type(inputs.type())
                gst_outputs_att = self.gst_layer.style_token_layer.attention(query, key)
                gst_outputs = gst_outputs + gst_outputs_att * v_amplifier
        elif style_input is None:
            gst_outputs = torch.zeros(1, 1, self.gst_embedding_dim).to(device)
        else:
            gst_outputs = self.gst_layer(style_input, speaker_embedding)  # pylint: disable=not-callable

        gst_outputs = gst_outputs.type(inputs.type())

        inputs = self._concat_speaker_embedding(inputs, gst_outputs)
        return inputs

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs


def load_model(hparams):
    model = Tacotron2(hparams)
    return model
