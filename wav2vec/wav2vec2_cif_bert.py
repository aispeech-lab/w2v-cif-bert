# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import contextlib
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from transformers import BertForMaskedLM

from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    GradMultiply,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransposeLast,
    Fp32LayerNorm,
    Fp32GroupNorm,
    FairseqDropout
)

from .wav2vec2_ctc import (
    Linear,
    Wav2VecEncoder,
    add_common_args,
    base_architecture
)
from .wav2vec2_cif import (
    CIFFcModel,
    cif_architecture,
)


def padding2attention_mask(padding_mask):

    mask1 = F.pad(padding_mask, [0, 1, 0, 0], value=1)
    mask2 = F.pad(padding_mask, [1, 0, 0, 0], value=0)
    mask = 1 - mask1.int() * mask2.int()

    return F.pad(mask, [1, 0, 0, 0], value=1)


def pred2bert_input(pred, token_mask, cls=101, sep=102):

    pred *= token_mask
    end_index = token_mask.sum(-1).long().unsqueeze(1) + 1
    pred.scatter_(dim=-1, index=end_index, value=sep)
    pred[:, 0] = cls

    return pred


def add_lm_args(parser):
    parser.add_argument(
        "--freeze-lm-finetune-updates", type=int, default=0, help="freeze_lm_finetune_updates"
    )
    parser.add_argument(
        "--gold-rate-range", type=str, help="gold-rate-range"
    )
    parser.add_argument(
        "--gold-rate-steps", type=str, help="gold-rate-steps"
    )
    parser.add_argument(
        "--infer-threash", type=float, default=0.8, help="infer-threash"
    )
    parser.add_argument(
        "--lambda-embedding", type=float, metavar="D", help="lambda-embedding"
    )
    parser.add_argument(
        "--lambda-am", type=float, default=1.0, metavar="D", help="lambda-am"
    )
    parser.add_argument(
        "--lambda-lm", type=float, default=0.2, metavar="D", help="lambda-lm"
    )
    parser.add_argument("--lambda-qua", type=float, default=0.1, metavar="D", help="lambda-qua")


@register_model("w2v_cif_bert")
class W2V_CIF_BERT(BaseFairseqModel):

    def __init__(self, args, encoder, bert, to_vocab, tgt_dict):
        """
        .copy_() clone to_vocab
        """
        super().__init__()
        self.encoder = encoder
        self.bert = bert
        self.dim_bert = bert.embeddings.word_embeddings.weight.size(1)
        self.to_vocab = to_vocab # 768 -> 21128
        self.to_vocab_ac = copy.deepcopy(to_vocab)
        self.to_vocab_ctc = copy.deepcopy(to_vocab)
        self.proj = Linear(encoder.d-1, self.dim_bert)
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.gold_rate_range = eval(args.gold_rate_range)
        self.gold_rate_steps = eval(args.gold_rate_steps)

        for p in self.bert.embeddings.parameters():
            p.requires_grad = False

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_lm_args(parser)
        parser.add_argument("--lambda-ctc", type=float, metavar="D", help="lambda-ctc")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)
        tgt_dict = task.target_dictionary

        bert, to_vocab = cls.build_bert(args, tgt_dict)
        encoder = cls.build_encoder(args) # encoder

        return cls(args, encoder, bert, to_vocab, tgt_dict)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_bert(cls, args, tgt_dict):
        pretrained_model = BertForMaskedLM.from_pretrained(args.bert_name)
        bert = pretrained_model.bert
        to_vocab = pretrained_model.cls

        return bert, to_vocab

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
        alphas = CIFFcModel.get_alphas(encoder_output)

        if self.training:
            gold_rate = self.set_gold_rate()
            decode_length = kwargs['target_lengths']
            gold_ids = kwargs['bert_input'].long()
            noise = 0.0
        else:
            gold_rate = 0.0
            decode_length = torch.round(alphas.sum(-1)).int()
            gold_ids = None
            noise = 0.0

        _alphas, num_output = self.resize(alphas, decode_length, noise=noise)
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        cif_outputs = self.cif(hidden_encoded, _alphas)
        hidden_ac = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden_ac)

        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            logits_lm, gold_embedding, pred_mask, token_mask = self.bert_forward(
                hidden_ac, logits_ac, padding_mask, gold_ids, gold_rate,
                threash=self.args.infer_threash)
        logits = self.args.lambda_am * logits_ac + self.args.lambda_lm * logits_lm
        logits *= (~padding_mask).unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output, 'gold_rate': gold_rate,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc,
                'pred_mask': pred_mask[:, 1:-1], 'token_mask': token_mask[:, 1:-1]}

    def bert_forward(self, hidden, logits_ac, padding_mask, gold_ids=None, gold_rate=0.0, threash=0.8):
        """
        """
        device = hidden.device
        token_mask = F.pad(~padding_mask, [1, 1, 0, 0], value=0)

        if self.training:
            input_ids = gold_ids
            pred_mask = (torch.rand(input_ids.size(), device=device) > gold_rate) * token_mask
        else: # infer
            probs = F.pad(utils.softmax(logits_ac.float(), dim=-1), [0, 0, 1, 1, 0, 0], value=0)
            confident, preds = probs.max(-1)
            input_ids = pred2bert_input(preds, token_mask)
            pred_mask = (confident <= threash) * token_mask

        # mixing
        gold_embedding = self.bert.embeddings.word_embeddings(input_ids)
        hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                 F.pad(hidden, [0, 0, 1, 1, 0, 0], value=0),
                                 gold_embedding)

        attention_mask = padding2attention_mask(padding_mask)
        embeddings = self.bert.embeddings(inputs_embeds=hidden_mix)
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask[:, None, None, :])

        logits = self.to_vocab(encoder_outputs[0])
        logits = logits[:, 1:-1, :]

        return logits, gold_embedding, pred_mask, token_mask

    @staticmethod
    def resize(*args, **kwargs):
        return CIFFcModel.resize(*args, **kwargs)

    @staticmethod
    def cif(*args, **kwargs):
        return CIFFcModel.cif(*args, **kwargs)

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

    def set_gold_rate(self):
        s, e = self.gold_rate_range
        s1, s2 = self.gold_rate_steps
        gold_rate = max((1 - max((self.num_updates - s1), 0) / s2) * (s-e), 0) + e

        return gold_rate


@register_model_architecture("w2v_cif_bert", "w2v_cif_bert")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.share_final_proj = getattr(args, "share_final_proj", False)
