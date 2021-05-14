# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from .wav2vec2_ctc import (
    Linear,
    Wav2VecEncoder,
    add_common_args,
    base_architecture
)
from .wav2vec2_cif import (
    get_alphas,
    cif,
    resize
)
from .wav2vec2_seq2seq import (
    TransformerDecoder,
    add_decoder_args,
    build_embedding,
    seq2seq_architecture,
    EncoderOut
)


@register_model("w2v_nar")
class W2V_NAR(BaseFairseqModel):

    def __init__(self, args, encoder, decoder):
        """
        .copy_() clone to_vocab
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_dict = decoder.dictionary
        self.to_vocab_ctc = Linear(encoder.d, len(decoder.dictionary))
        self.proj = Linear(encoder.d-1, args.decoder_embed_dim)
        self.num_updates = 0
        self.args = args

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        parser.add_argument("--lambda-qua", type=float, metavar="D", help="lambda-qua")
        parser.add_argument("--lambda-ctc", type=float, metavar="D", help="lambda-ctc")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_nar_architecture(args)
        tgt_dict = task.target_dictionary

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        encoder = cls.build_encoder(args) # encoder
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerNonAutoRegressiveDecoder(args, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        """
        encoder_output= "encoder_out": x,
                        "encoded": encoded,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
        """
        encoder_output = self.encoder(tbc=False, **kwargs)
        hidden_encoded = encoder_output['encoder_out'][:, :, :-1]
        hidden_ctc = F.pad(hidden_encoded, [0, 1, 0, 0, 0, 0], value=0)
        logits_ctc = self.to_vocab_ctc(hidden_ctc)
        len_logits_ctc = (~encoder_output['padding_mask']).sum(-1).long()
        alphas = get_alphas(encoder_output)
        decode_length = kwargs['target_lengths'] if self.training else torch.round(alphas.sum(-1)).int()
        _, num_output = self.resize(alphas, decode_length)

        padding_mask = ~utils.sequence_mask(decode_length).bool()
        token_mask = ~padding_mask
        mask_ids = torch.ones_like(padding_mask) * self.tgt_dict.bos()
        # emcoded = self.proj(hidden_encoded)
        encoder_out = EncoderOut(
            encoder_out=hidden_ctc.transpose(0, 1),  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_output['encoder_padding_mask'],  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

        if self.training:
            gold_ids = kwargs['target'].long()
            rand = torch.rand(gold_ids.size(), device=gold_ids.device) * token_mask
            list_pred_mask = []
            for i, l in enumerate(decode_length):
                k = random.randint(1, l)
                list_pred_mask.append(rand[i] >= torch.topk(rand[i], k).values.min())
            pred_mask = torch.stack(list_pred_mask, 0) * token_mask
            gold_mask = ~pred_mask * token_mask
            gold_rate = gold_mask.sum() * 1.0 / token_mask.sum()
            decoder_input_ids = torch.where(pred_mask, mask_ids, gold_ids)
            logits = self.decoder(encoder_out=encoder_out,
                                  prev_output_tokens=decoder_input_ids)
            # import pdb; pdb.set_trace()
        else:
            pred_mask = gold_rate = 0.0
            decoder_input_ids = mask_ids
            for _ in range(10):
                logits = self.decoder(encoder_out=encoder_out,
                                      prev_output_tokens=decoder_input_ids)
                probs, pred_ids = utils.softmax(logits, dim=-1).max(-1)
                gold_mask = probs > 0.9
                decoder_input_ids = torch.where(gold_mask, pred_ids, mask_ids) * token_mask

        logits *= token_mask.unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length, 'gold_rate': gold_rate,
                'alphas': alphas, 'num_output': num_output, 'pred_mask': pred_mask,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc}

    def get_normalized_probs(self, net_output, log_probs, retrun_ctc=False):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits_ctc = net_output["logits_ctc"]
        logits = net_output["logits"]
        if log_probs:
            res_ctc = utils.log_softmax(logits_ctc.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res_ctc = utils.softmax(logits_ctc.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)
        res_ctc.batch_first = True
        res.batch_first = True

        if retrun_ctc:
            return res_ctc, res
        else:
            return res

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model("w2v_cif_nar")
class W2V_CIF_NAR(W2V_NAR):

    def forward(self, **kwargs):
        """
        encoder_output= "encoder_out": x,
                        "encoded": encoded,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
        """
        encoder_output = self.encoder(tbc=False, **kwargs)
        hidden_encoded = encoder_output['encoder_out'][:, :, :-1]
        hidden_ctc = F.pad(hidden_encoded, [0, 1, 0, 0, 0, 0], value=0)
        logits_ctc = self.to_vocab_ctc(hidden_ctc)
        len_logits_ctc = (~encoder_output['padding_mask']).sum(-1).long()
        alphas = get_alphas(encoder_output)
        decode_length = kwargs['target_lengths'] if self.training else torch.round(alphas.sum(-1)).int()
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        _alphas, num_output = self.resize(alphas, decode_length)

        # if not self.training:
        #     import pdb; pdb.set_trace()
        encoder_out = EncoderOut(
            encoder_out=encoder_output['encoder_out'].transpose(0, 1),  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_output['encoder_padding_mask'],  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )
        prev_output_tokens = torch.ones_like(padding_mask) * self.tgt_dict.bos()
        decoder_out = self.decoder(encoder_out=encoder_out,
                                   prev_output_tokens=prev_output_tokens)
        logits = decoder_out["logits"]
        logits *= (~padding_mask).unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc}


class TransformerNonAutoRegressiveDecoder(TransformerDecoder):

    def forward(self, prev_output_tokens, encoder_out=None, **unused):
        prev_output_tokens = prev_output_tokens.long()

        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, _, _ = layer(
                    x,
                    encoder_out.encoder_out,
                    encoder_out.encoder_padding_mask,
                    self_attn_mask=None,
                )

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.output_layer(x)

        return x


@register_model_architecture("w2v_nar", "w2v_nar")
def w2v_cif_nar_architecture(args):
    seq2seq_architecture(args)


@register_model_architecture("w2v_cif_nar", "w2v_cif_nar")
def w2v_cif_nar_architecture(args):
    seq2seq_architecture(args)
