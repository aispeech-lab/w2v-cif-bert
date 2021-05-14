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
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransposeLast,
    Fp32LayerNorm,
    FairseqDropout,
    Fp32GroupNorm
)

from .wav2vec2_ctc import add_common_args, Wav2VecEncoder, Linear, base_architecture

PAD_IDX = 1
EOS_IDX = 2
eps = 1e-7
THRESHOLD = 0.95


def add_decoder_args(parser):
    parser.add_argument(
        "--decoder-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension",
    )
    parser.add_argument(
        "--decoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--decoder-layers", type=int, metavar="N", help="num decoder layers"
    )
    parser.add_argument(
        "--decoder-layerdrop",
        type=float,
        metavar="D",
        help="decoder layerdrop chance",
    )
    parser.add_argument(
        "--decoder-attention-heads",
        type=int,
        metavar="N",
        help="num decoder attention heads",
    )
    parser.add_argument(
        "--decoder-learned-pos",
        action="store_true",
        help="use learned positional embeddings in the decoder",
    )
    parser.add_argument(
        "--decoder-normalize-before",
        action="store_true",
        help="apply layernorm before each decoder block",
    )
    parser.add_argument(
        "--no-token-positional-embeddings",
        default=False,
        action="store_true",
        help="if set, disables positional embeddings (outside self attention)",
    )

    parser.add_argument(
        "--decoder-dropout",
        type=float,
        metavar="D",
        help="dropout probability in the decoder",
    )
    parser.add_argument(
        "--decoder-attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside the decoder",
    )
    parser.add_argument(
        "--decoder-activation-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside the decoder",
    )
    parser.add_argument(
        "--teacher-forcing-updates", type=int, help="is teacher-forcing"
    )


@register_model("wav2vec_seq2seq")
class TransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        seq2seq_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)

        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["logits"]
        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)
        res.batch_first = True

        return res

    def get_encoder_output(self, net_input):
        encoder_out = self.encoder(tbc=True, **net_input)
        return EncoderOut(
            encoder_out=encoder_out['encoder_out'],  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_out['encoder_padding_mask'],  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


@register_model("wav2vec_ctc_seq2seq")
class TransformerCTCModel(TransformerModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        seq2seq_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args, tgt_dict)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        ctc_logits = encoder_out["encoder_out"]
        encoder_out["encoder_out"] = encoder_out["encoded"]
        logits, _ = self.decoder(encoder_out=encoder_out, **kwargs)
        encoder_out["ctc_logits"] = ctc_logits

        return encoder_out, logits

    def get_normalized_probs(self, ctc_logits, logits, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        if log_probs:
            ctc_res = utils.log_softmax(ctc_logits.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            ctc_res = utils.softmax(ctc_logits.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)
        ctc_res.batch_first = True
        res.batch_first = True

        return ctc_res, res


class TransformerDecoder(FairseqIncrementalDecoder):
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

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        self.dropout = args.decoder_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        args.encoder_embed_dim = embed_dim

        self.layerdrop = args.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings else None
        )

        args = copy.deepcopy(args)
        args.dropout = args.decoder_dropout
        args.attention_dropout = args.decoder_attention_dropout
        args.activation_dropout = args.decoder_activation_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        # x = self.output_layer(encoder_out.encoder_out.transpose(0,1) + x)
        x = self.output_layer(x)
        extra["logits"] = x

        return extra

    def extract_features(
        self, prev_output_tokens, encoder_output=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_output.encoder_out if encoder_output is not None else None,
                    encoder_output.encoder_padding_mask if encoder_output is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    return emb


@register_model_architecture("wav2vec_seq2seq", "wav2vec_seq2seq")
def seq2seq_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 10)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.decoder_attention_dropout = getattr(args, "share-decoder-input-output-embed", 0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    base_architecture(args)


@register_model_architecture("wav2vec_ctc_seq2seq", "wav2vec_ctc_seq2seq")
def ctc_seq2seq_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.decoder_attention_dropout = getattr(args, "share-decoder-input-output-embed", 0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    base_architecture(args)
