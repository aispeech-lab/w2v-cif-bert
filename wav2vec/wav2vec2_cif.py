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
from .wav2vec2_ctc import add_common_args, Wav2VecEncoder, Linear, base_architecture

PAD_IDX = 1
EOS_IDX = 2
eps = 1e-7
THRESHOLD = 0.999


def add_cif_args(parser):
    parser.add_argument("--lambda-qua", type=float, metavar="D", help="lambda-qua")
    parser.add_argument("--lambda-alpha", type=float, metavar="D", help="lambda-alpha")


@register_model("wav2vec_cif_fc")
class CIFFcModel(BaseFairseqModel):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
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


@register_model_architecture("wav2vec_cif_fc", "wav2vec_cif_fc")
def cif_architecture(args):
    args.lambda_qua = getattr(args, "lambda_qua", 0.05)
    args.lambda_alpha = getattr(args, "lambda_alpha", 0.1)
    base_architecture(args)
