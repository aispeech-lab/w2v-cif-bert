# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import utils, checkpoint_utils, tasks
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransposeLast,
    Fp32LayerNorm,
    FairseqDropout,
    Fp32GroupNorm
)

from .wav2vec2 import conv_1d_block
from .wav2vec2_ctc import add_common_args, Wav2VecEncoder, Linear, base_architecture

PAD_IDX = 1
EOS_IDX = 2
eps = 1e-7
THRESHOLD = 0.999


def add_cif_args(parser):
    parser.add_argument(
        "--assigner-conv-layers",
        type=str,
        metavar="EXPR",
        help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
    )
    parser.add_argument(
        "--extractor-mode",
        type=str,
        metavar="EXPR",
        help="extractor_mode",
    )
    parser.add_argument("--lambda-qua", type=float, metavar="D", help="lambda-qua")
    parser.add_argument("--lambda-alpha", type=float, metavar="D", help="lambda-alpha")


@register_model("wav2vec_cif_fc")
class CIFFcModel(BaseFairseqModel):
    def __init__(self, args, encoder, assigner, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.assigner = assigner
        self.decoder = decoder

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_cif_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        cif_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        decoder = cls.build_decoder(args, encoder.d, tgt_dict)

        return cls(args, encoder, assigner, decoder)

    @classmethod
    def build_encoder(cls, args):
        return Wav2VecEncoder(args)

    @classmethod
    def build_assigner(cls, args, dim_input):
        return Assigner(args, dim_input)

    @classmethod
    def build_decoder(cls, args, input_dim, tgt_dict):
        return FCDecoder(args, input_dim, tgt_dict)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas, alphas_pen = self.assigner(encoder_output)
        if self.training:
            _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        else:
            _alphas, num_output = self.resize(alphas)
        cif_outputs = self.cif(encoder_output, _alphas)

        if self.training and cif_outputs.abs().sum(-1).ne(0).sum() != kwargs['target_lengths'].sum():
            print('_alphas:\t', _alphas.sum(-1))
            print('alphas:\t', alphas.sum(-1))
            print('target:\t', kwargs['target_lengths'])
            import pdb; pdb.set_trace()

        logits = self.decoder(cif_outputs)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output, 'alphas_pen': alphas_pen}

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["logits"]
        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)

        res.batch_first = True

        return res

    @staticmethod
    def cif(encoder_output, alphas, threshold=THRESHOLD, log=False):
        if type(encoder_output) is Tensor:
            hidden = encoder_output
        elif 'encoded' in encoder_output.keys():
            hidden = encoder_output['encoded']
        else:
            hidden = encoder_output['encoder_out']

        device = hidden.device
        B, T, H = hidden.size()

        # loop varss
        integrate = torch.zeros([B], device=device)
        frame = torch.zeros([B, H], device=device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(T):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([B], device=device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= threshold
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([B], device=device),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, H),
                                remainds[:, None] * hidden[:, t, :],
                                frame)

            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f} fire: {}'.format(
                    t, integrate[log], cur[log], remainds[log], fire_place[log]))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(B):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire >= threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
            list_ls.append(torch.cat([l, pad_l], 0))

            if log:
                print(b, l.size(0))

        if log:
            print('fire:\n', fires[log])
            print('fire place:\n', torch.where(fires[log] >= threshold))

        return torch.stack(list_ls, 0)

    # @staticmethod
    # def cif(encoder_outputs, not_padding, targets=None):
    #     if 'encoded' in encoder_output.keys():
    #         hidden = encoder_output['encoded']
    #     else:
    #         hidden = encoder_output['encoder_out']
    #     device = hidden.device
    #     B, T, H = hidden.size()
    #
    #     accumulated_weights = torch.zeros([B, 0]).to(device)
    #     accumulated_states = torch.zeros([B, 0, H]).to(device)
    #     fired_states = torch.zeros([B, 0, H]).to(device)
    #     # def cif_inner_loop(i, accumulated_weights, accumulated_states, fired_states):
    #     for i in range(T):
    #         # get previous states from the recorded tensor
    #         if i == 0:
    #             prev_accumulated_weight = torch.zeros([B]).to(device)
    #             prev_accumulated_state = torch.zeros([B, H]).to(device)
    #         else:
    #             prev_accumulated_weight = accumulated_weights[:, i-1]
    #             prev_accumulated_state = accumulated_states[:, i-1, :]
    #
    #         # decide whether positioning a boundary
    #         cur_is_fired = prev_accumulated_weight + a[:, i] > threshold
    #
    #         # update the accumulated weights by considering whether positioning a boundary
    #         cur_weight =a[:, i, None]
    #         prev_accumulated_weight = prev_accumulated_weight[:, None]
    #         remained_weight = 1.0 - prev_accumulated_weight
    #
    #         cur_accumulated_weight = torch.where(
    #             cur_is_fired,
    #             cur_weight - remained_weight,
    #             cur_weight + prev_accumulated_weight)
    #
    #         cur_accumulated_state = torch.where(
    #             cur_is_fired,
    #             (cur_weight - remained_weight) * encoder_outputs[:, i, :],
    #             prev_accumulated_state + cur_weight * encoder_outputs[:, i, :])
    #
    #         cur_fired_state = torch.where(
    #             cur_is_fired,
    #             prev_accumulated_state + remained_weight * encoder_outputs[:, i, :],
    #             torch.zeros([B, H]).to(device))
    #
    #         cur_fired_state = torch.where(
    #             torch.fill([B], i) > first_padding_pos,
    #             torch.zeros([B, H]),
    #             cur_fired_state)
    #
    #         accumulated_weights = tf.concat([accumulated_weights, cur_accumulated_weight], axis=1)
    #         accumulated_states = tf.concat([accumulated_states,
    #                                         tf.expand_dims(cur_accumulated_state, axis=1)], axis=1)
    #         fired_states = tf.concat([fired_states,
    #                                   tf.expand_dims(cur_fired_state, axis=1)], axis=1)
    #
    #     fired_marks = tf.to_int32(tf.not_equal(tf.reduce_sum(tf.abs(fired_states), axis=-1), 0.0))
    #
    #
    #     fired_utt_length = tf.count_nonzero(fired_marks, axis=-1)
    #     # fired_utt_length = tf.Print(fired_utt_length, [fired_utt_length], message='fired_utt_length', summarize=202)
    #     fired_max_length = tf.to_int32(tf.reduce_max(fired_utt_length))
    #     # fired_max_length = tf.Print(fired_max_length, [fired_max_length], summarize=202)
    #
    #     def extract_for_each_utt(j, cif_outputs):
    #         cur_utt_fired_mark = fired_marks[j, :]
    #         cur_utt_fired_state = fired_states[j, :, :]
    #
    #         cur_utt_outputs = tf.dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
    #         cur_utt_output = cur_utt_outputs[1]
    #
    #         cur_utt_length = common_layers.shape_list(cur_utt_output)[0]
    #         pad_length = fired_max_length - cur_utt_length
    #         cur_utt_output = tf.concat([cur_utt_output,
    #                                     tf.fill([pad_length, hidden_size], 0.0)], axis=0)
    #         cur_utt_output = tf.expand_dims(cur_utt_output, axis=0)
    #         cif_outputs = tf.concat([cif_outputs, cur_utt_output], axis=0)
    #
    #         return j+1, cif_outputs
    #
    #     init_cif_outputs = tf.zeros([0, fired_max_length, hidden_size])
    #     _, cif_outputs = tf.while_loop(
    #         lambda j, *_: tf.less(j, batch_size),
    #         extract_for_each_utt,
    #         [tf.constant(0), init_cif_outputs],
    #         shape_invariants=[
    #             tf.TensorShape([]),
    #             tf.TensorShape([None, None, hidden_size])
    #         ]
    #     )
    #
    #     # calculate the not_padding according to the cif_outputs
    #     not_padding_after_cif = tf.to_int32(tf.not_equal(tf.reduce_sum(tf.abs(cif_outputs), axis=-1), 0.0))
    #
    #     # for the calculation of num char
    #     if is_training and hparams.use_scaling_strategy:
    #         sum_a = tf.reduce_sum(a_org, axis=1)
    #     else:
    #         sum_a = tf.reduce_sum(a, axis=1)
    #
    #     return cif_outputs, not_padding_after_cif, sum_a

    @staticmethod
    def resize(alphas, target_lengths, noise=0.0, threshold=THRESHOLD):
        """
        alpha in thresh=1.0 | (0.0, +0.21)
        target_lengths: if None, apply round and resize, else apply scaling
        """
        device = alphas.device
        # sum
        _num = alphas.sum(-1)

        num = target_lengths.float()
        num = num + noise * torch.rand(alphas.size(0)).to(device)

        # scaling
        _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))

        # rm attention value that exceeds threashold
        while len(torch.where(_alphas > threshold)[0]):
            print('fixing alpha')
            xs, ys = torch.where(_alphas > threshold)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).float()
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask

        return _alphas, _num


@register_model("wav2vec_cif_fc_v2")
class CIFFcModelV2(BaseFairseqModel):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        parser.add_argument("--lambda-qua", type=float, metavar="D", help="lambda-qua")
        parser.add_argument("--lambda-alpha", type=float, metavar="D", help="lambda-alpha")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        cif_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, encoder.d, tgt_dict)

        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return Wav2VecEncoder(args)

    @classmethod
    def build_decoder(cls, args, input_dim, tgt_dict):
        return FCDecoder(args, input_dim, tgt_dict)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = self.get_alphas(encoder_output)
        if self.training:
            _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        else:
            _alphas, num_output = self.resize(alphas)
        cif_outputs = self.cif(encoder_output, _alphas)

        if self.training and cif_outputs.abs().sum(-1).ne(0).sum() != kwargs['target_lengths'].sum():
            print('_alphas:\t', _alphas.sum(-1))
            print('alphas:\t', alphas.sum(-1))
            print('target:\t', kwargs['target_lengths'])
            import pdb; pdb.set_trace()

        logits = self.decoder(cif_outputs)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output}

    @staticmethod
    def get_alphas(encoder_output):
        padding_mask = encoder_output['padding_mask']

        if 'encoded' in encoder_output.keys():
            alphas = encoder_output['encoded'][:, :, -1]
        else:
            alphas = encoder_output['encoder_out'][:, :, -1]

        alphas = torch.sigmoid(alphas)
        alphas = alphas * (~padding_mask).float()

        return alphas

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["logits"]
        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)

        res.batch_first = True

        return res

    @staticmethod
    def cif(*args, **kwargs):
        return CIFFcModel.cif(*args, **kwargs)

    @staticmethod
    def resize(*args, **kwargs):
        return CIFFcModel.resize(*args, **kwargs)


class Assigner(FairseqEncoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~dataload.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dim_input):
        super().__init__()
        assigner_conv_layers = eval(args.assigner_conv_layers)
        self.embed = assigner_conv_layers[-1][0]
        self.feature_extractor = Conv1DFeatureExtractionModel(
            dim_input=dim_input,
            conv_layers=assigner_conv_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=True,
            output='same'
        )
        self.proj = Linear(self.embed, 1)

    def forward(self, encoder_output):
        """
        Args:
            encoder_out (FloatTensor): previous decoder outputs of shape
                `(B, T, H)`, for teacher forcing
            encoded_lengths (Tensor): output from the encoder, used for
                encoder-side attention
        Returns:
            the decoder's output of shape `(batch, src_len)`
        """
        if 'encoded' in encoder_output.keys():
            encoded, padding_mask = encoder_output['encoded'], encoder_output['padding_mask']
        else:
            encoded, padding_mask = encoder_output['encoder_out'], encoder_output['padding_mask']

        x = self.feature_extractor(encoded)
        x = self.proj(x)[:, :, 0]
        features_pen = x.pow(2).mean()
        x = torch.sigmoid(x)
        x = x * (~padding_mask).float()

        return x, features_pen


class FCDecoder(FairseqEncoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~dataload.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, input_dim, dictionary):
        super().__init__(dictionary)
        self.proj = Linear(input_dim, len(dictionary), bias=True)

    def forward(self, encoded):
        """
        Args:
            encoder_out (Tensor): output from the encoder, used for
                encoder-side attention
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
        """
        x = self.proj(encoded)
        return x


class Conv1DFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        dim_input: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        output: str = "valid", # ["valid", "same"]
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}
        assert output in {"valid", "same"}
        self.output = output

        in_d = dim_input
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                conv_1d_block(in_d, dim, k, stride, dropout=0.0,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias)
            )
            in_d = dim

    def forward(self, x):
        if self.output == 'same':
            length = x.size(1)
            x = F.pad(x, [0, 0, 0, 10, 0, 0])
        x = x.transpose(1, 2)

        for conv in self.conv_layers:
            x = conv(x)

        x = x.transpose(1, 2)

        if self.output == 'same':
            x = x[:, :length, :]

        return x


@register_model_architecture("wav2vec_cif_fc", "wav2vec_cif_fc")
def cif_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", 'default')
    args.conv_bias = getattr(args, "conv-bias", False)
    args.lambda_qua = getattr(args, "lambda_qua", 0.05)
    args.lambda_alpha = getattr(args, "lambda_alpha", 0.1)
    base_architecture(args)


@register_model_architecture("wav2vec_cif_fc_v2", "wav2vec_cif_fc_v2")
def cif_v2_architecture(args):
    cif_architecture(args)
