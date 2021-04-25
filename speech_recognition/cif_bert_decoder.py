#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
CTC decoders.
"""
import itertools as it
import torch

from fairseq import utils
from fairseq.models.wav2vec.wav2vec2_cif import CIFFcModel


class CIF_BERT_Decoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest
        self.infer_threshold = args.infer_threshold

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        model = models[0]

        # encoder_output = model.encoder(tbc=False, **sample["net_input"])
        # alphas = CIFFcModel.get_alphas(encoder_output)
        # decode_length = torch.round(alphas.sum(-1)).int()
        # _alphas, num_output = model.resize(alphas, decode_length, noise=0.0)
        #
        # padding_mask = ~utils.sequence_mask(decode_length).bool()
        # cif_outputs = model.cif(encoder_output['encoder_out'][:, :, :-1], _alphas)
        # hidden = model.proj(cif_outputs)
        # logits_ac = model.to_vocab_ac(hidden)
        #
        # infer_threash = self.infer_threshold if self.infer_threshold else model.args.infer_threash
        # for i in range(1):
        #     logits, gold_embedding, pred_mask, token_mask = model.bert_forward(
        #         hidden, logits_ac, padding_mask, None, 0.0,
        #         threash=infer_threash)
        #     logits = self.args.lambda_am * logits_ac + model.args.lambda_lm * logits
        # probs = utils.softmax(logits.float(), dim=-1)
        net_output = model(**sample["net_input"])
        logits = net_output['logits']
        probs = utils.softmax(logits.float(), dim=-1)
        decode_length = net_output['len_logits']

        res = []
        for distribution, length in zip(probs, decode_length):
            result = distribution.argmax(-1)
            score = 0.0
            res.append([{'tokens': result[:length],
                         "score": score}])

        return res

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))
