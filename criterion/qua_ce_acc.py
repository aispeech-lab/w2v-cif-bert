# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import random

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

from .cross_entropy_acc import LabelSmoothedCrossEntropyWithAccCriterion

random.seed(0)


def logits2sent(preds, targets, dictionary, rate=0.03):
    if random.random() < rate:
        try:
            pred = dictionary.tokenizer.decode(preds)
            target = dictionary.tokenizer.decode(targets)
        except:
            pred = dictionary.string(preds)
            target = dictionary.string(targets)
        print('pred:\n{}\ntarget:\n{}\n'.format(pred, target))


@register_criterion("qua_ce_acc")
class QuantityCrossEntropyWithAccCriterion(LabelSmoothedCrossEntropyWithAccCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.decoder = self.build_decoder(args, task)

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def build_decoder(self, args, task):
        decoder = getattr(args, "decoder", None)

        from examples.speech_recognition.cif_decoder import CIFDecoder
        if decoder == "cif_decoder":

            decoder = CIFDecoder(args, task.target_dictionary, {})
        elif decoder == "cif_lm_decoder":

            decoder = CIFDecoder(args, task.target_dictionary, ({}, {}))
        else:
            import pdb; pdb.set_trace()

        return decoder

    def compute_loss(self, model, net_output, sample, reduction, log_probs):
        # number loss
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2) + 1e-6).sum()
        qua_loss = diff
        # alphas_pen
        # alphas_pen = net_output["alphas_pen"]
        # qua_loss = diff + self.args.lambda_alpha * alphas_pen
        target = sample["target"]  # no eos bos
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduction,
        )

        return lprobs, qua_loss, ce_loss

    def get_logging_output(self, sample, lprobs, loss, qua_loss, ce_loss):
        target = sample["target"].view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        net_output = model(**sample["net_input"])
        lprobs, qua_loss, ce_loss = self.compute_loss(
            model, net_output, sample, reduction, log_probs
        )

        nsentences = sample["target"].size(0) + 1.0
        ntokens = sample["ntokens"]
        loss = self.args.lambda_qua * qua_loss * ntokens / nsentences + ce_loss

        sample_size, logging_output = self.get_logging_output(
            sample, lprobs, loss, qua_loss, ce_loss
        )

        if not model.training:
            import editdistance

            c_err = 0
            c_len = 0
            self.decoder.step_forward_fn = model.decoder
            with torch.no_grad():
                decodeds = self.decoder.generate([model], sample)
                for decoded, t in zip(decodeds, sample["target"]):
                    decoded = decoded[0]['tokens']

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss = sum(log['ce_loss'] for log in logging_outputs)
        qua_loss = sum(log['qua_loss'] for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
            "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = ce_loss / ntokens
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("qua_ce_acc_v2")
class QuantityCrossEntropyWithAccCriterionV2(LabelSmoothedCrossEntropyWithAccCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def compute_loss(self, model, net_output, sample, reduction, log_probs):
        # number loss
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2) + 1e-6).sum()
        qua_loss = diff
        # alphas_pen
        # alphas_pen = net_output["alphas_pen"]
        # qua_loss = diff + self.args.lambda_alpha * alphas_pen
        target = sample["target"]  # no eos bos
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduction,
        )

        return lprobs, qua_loss, ce_loss

    def get_logging_output(self, sample, lprobs, loss, qua_loss, ce_loss):
        target = sample["target"].view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        net_output = model(**sample["net_input"])
        num_output = net_output["num_output"].int()

        if model.training:
            lprobs, qua_loss, ce_loss = self.compute_loss(
                model, net_output, sample, reduction, log_probs
            )

            nsentences = sample["target"].size(0) + 1.0
            ntokens = sample["ntokens"]
            loss = self.args.lambda_qua * qua_loss * ntokens / nsentences + ce_loss

            sample_size, logging_output = self.get_logging_output(
                sample, lprobs, loss, qua_loss, ce_loss
            )
        else:
            import editdistance

            loss = qua_loss = sample_size = 0.0
            logging_output = {
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size
            }
            c_err = 0
            c_len = 0
            with torch.no_grad():
                for logits, l, t in zip(net_output['logits'], num_output, sample["target"]):
                    decoded = logits.argmax(dim=-1)[:l]
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()
                    # targ_units_arr = targ.unique_consecutive().tolist()
                    # pred_units_arr = decoded.unique_consecutive().tolist()
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss = sum(log.get("ce_loss", 0) for log in logging_outputs)
        qua_loss = sum(log.get("qua_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
            "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = ce_loss / ntokens
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("nar_qua_ctc_ce")
class NAR_QUA_CTC_CE(QuantityCrossEntropyWithAccCriterionV2):

    def compute_loss(self, model, net_output, sample, reduction, log_probs):

        pad_id = self.task.target_dictionary.pad()
        # number loss
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2) + 1e-12).sum()
        qua_loss = diff
        lprobs_ctc, lprobs = model.get_normalized_probs(net_output, retrun_ctc=True, log_probs=log_probs)

        # CE loss
        pred_mask = net_output["pred_mask"]
        lprobs = lprobs.view(-1, lprobs.size(-1)) # N, T, D -> N * T, D
        target_masked = torch.where(
            pred_mask,
            sample["target"],
            torch.ones_like(sample["target"]) * pad_id).long().view(-1)
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs,
            target_masked,
            0.0,
            ignore_index=pad_id,
            reduce=reduction,
        )

        # CTC loss
        target = sample["target"]
        pad_mask = target != pad_id
        targets_flat = target.masked_select(pad_mask)
        target_lengths = sample["target_lengths"]

        len_lprobs = net_output["len_logits_ctc"]
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs_ctc.transpose(0, 1), # T x B x V
                targets_flat,
                len_lprobs,
                target_lengths,
                blank=self.task.target_dictionary.blk(),
                reduction="sum",
                zero_infinity=True,
            )

        return lprobs, target_masked, ctc_loss, qua_loss, ce_loss

    def forward(self, model, sample, reduction="sum", log_probs=True):
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]

        if model.training:
            net_output = model(**sample["net_input"])
            num_output = torch.round(net_output["num_output"]).int()
            gold_rate = net_output["gold_rate"]
            sample_size = net_output["pred_mask"].float().sum()
            lprobs, targets, ctc_loss, qua_loss, ce_loss = self.compute_loss(
                model, net_output, sample, reduction, log_probs
            )

            e_len = int(sum(abs(sample["target_lengths"].data - num_output.data)))
            loss = ce_loss + \
                   self.args.lambda_qua * qua_loss * sample_size / nsentences + \
                   self.args.lambda_ctc * ctc_loss * sample_size / ntokens

            sample_size, logging_output = self.get_logging_output(
                sample, lprobs, targets, e_len, loss, ctc_loss, qua_loss, ce_loss, gold_rate
            )
        else:
            import editdistance

            net_output = model(**sample["net_input"])
            num_output = torch.round(net_output["num_output"]).int()

            loss = sample_size = 0.0
            logging_output = {
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size
            }
            c_err = 0
            c_len = 0
            e_len = 0
            with torch.no_grad():
                for i, logits, l, t in zip(range(9999), net_output['logits'], num_output, sample["target"]):
                    # decoded = logits.argmax(dim=-1)[:l]
                    p = t != self.task.target_dictionary.pad()
                    decoded = logits.argmax(dim=-1)[:l]
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)
                    e_len += abs(len(targ_units_arr) - len(pred_units_arr)) * 1.0
                logits2sent(pred_units_arr, targ_units_arr, model.tgt_dict, rate=0.03)
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len
                logging_output["e_len"] = e_len

        return loss, sample_size, logging_output

    def get_logging_output(self, sample, lprobs, targets, e_len, loss, ctc_loss, qua_loss, ce_loss, gold_rate):
        mask = targets != 0
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == targets.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = max(utils.item(mask.sum().float().data), 1.0)

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
            "gold_rate": gold_rate,
            "ntokens": sample_size,
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "e_len": e_len
        }

        return sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        e_len = sum(log.get("e_len", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ctc_loss = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ce_loss = sum(log.get("ce_loss", 0) for log in logging_outputs)
        qua_loss = sum(log.get("qua_loss", 0) for log in logging_outputs)
        gold_rate = sum(log.get("gold_rate", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0: # training
            agg_output = {
                "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
                "ctc_loss": ctc_loss / ntokens if ntokens > 0 else 0.0,
                "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
                "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
                "e_len": e_len / nsentences if nsentences > 0 else 0.0,
                # if args.sentence_avg, then sample_size is nsentences, then loss
                # is per-sentence loss; else sample_size is ntokens, the loss
                # becomes per-output token loss
                "gold_rate": gold_rate / len(logging_outputs),
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size,
                "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
                "correct": correct_sum,
                "total": total_sum,
                # total is the number of validate tokens
            }
        else:
            c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
            c_total = sum(log.get("c_total", 1) for log in logging_outputs)
            agg_output = {
                "uer": c_errors * 100.0 / c_total,
                "qua_error": e_len * 100.0 / c_total,
                "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
                "gold_rate": gold_rate,
                "ntokens": ntokens,
                "nsentences": nsentences
            }

        return agg_output


@register_criterion("qua_ctc_ce")
class QUA_CTC_CE(QuantityCrossEntropyWithAccCriterionV2):

    def compute_loss(self, model, net_output, sample, reduction, log_probs):

        # number loss
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2) + 1e-12).sum()
        qua_loss = diff

        # N, T -> N * T
        target = sample["target"].view(-1)
        lprobs_ctc, lprobs = model.get_normalized_probs(net_output, retrun_ctc=True, log_probs=log_probs)

        lprobs = lprobs.view(-1, lprobs.size(-1))
        # N, T, D -> N * T, D
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs,
            target.long(),
            0.0,
            ignore_index=self.task.target_dictionary.pad(),
            reduce=reduction,
        )

        # CTC loss
        target = sample["target"]
        pad_mask = target != self.task.target_dictionary.pad()
        targets_flat = target.masked_select(pad_mask)
        target_lengths = sample["target_lengths"]

        len_lprobs = net_output["len_logits_ctc"]
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs_ctc.transpose(0, 1), # T x B x V
                targets_flat,
                len_lprobs,
                target_lengths,
                blank=self.task.target_dictionary.blk(),
                reduction="sum",
                zero_infinity=True,
            )

        return lprobs, ctc_loss, qua_loss, ce_loss

    def get_logging_output(self, sample, lprobs, e_len, loss, ctc_loss, qua_loss, ce_loss, gold_rate):
        targets = sample["target"].view(-1)
        mask = targets != 100
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == targets.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = max(utils.item(mask.sum().float().data), 1.0)

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
            "gold_rate": gold_rate,
            "ntokens": sample_size,
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "e_len": e_len
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]

        if model.training:
            net_output = model(**sample["net_input"])
            num_output = torch.round(net_output["num_output"]).int()
            gold_rate = net_output["gold_rate"] if "gold_rate" in net_output else 0.0
            lprobs, ctc_loss, qua_loss, ce_loss = self.compute_loss(
                model, net_output, sample, reduction, log_probs
            )

            e_len = int(sum(abs(sample["target_lengths"].data - num_output.data)))
            loss = ce_loss + \
                   self.args.lambda_qua * qua_loss * ntokens / nsentences + \
                   self.args.lambda_ctc * ctc_loss

            sample_size, logging_output = self.get_logging_output(
                sample, lprobs, e_len, loss, ctc_loss, qua_loss, ce_loss, gold_rate
            )
        else:
            import editdistance

            net_output = model(**sample["net_input"])
            num_output = torch.round(net_output["num_output"]).int()

            loss = sample_size = 0.0
            logging_output = {
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size
            }
            c_err = 0
            c_len = 0
            e_len = 0
            with torch.no_grad():
                for i, logits, l, t in zip(range(9999), net_output['logits'], num_output, sample["target"]):
                    # decoded = logits.argmax(dim=-1)[:l]
                    p = t != self.task.target_dictionary.pad()
                    decoded = logits.argmax(dim=-1)[:l]
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)
                    e_len += abs(len(targ_units_arr) - len(pred_units_arr)) * 1.0
                logits2sent(pred_units_arr, targ_units_arr, model.tgt_dict, rate=0.03)
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len
                logging_output["e_len"] = e_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        e_len = sum(log.get("e_len", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ctc_loss = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ce_loss = sum(log.get("ce_loss", 0) for log in logging_outputs)
        gold_rate = sum(log.get("gold_rate", 0) for log in logging_outputs)
        qua_loss = sum(log.get("qua_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0: # training
            agg_output = {
                "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
                "ctc_loss": ctc_loss / ntokens if ntokens > 0 else 0.0,
                "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
                "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
                "e_len": e_len / nsentences if nsentences > 0 else 0.0,
                # if args.sentence_avg, then sample_size is nsentences, then loss
                # is per-sentence loss; else sample_size is ntokens, the loss
                # becomes per-output token loss
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size,
                "gold_rate": gold_rate / len(logging_outputs),
                "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
                "correct": correct_sum,
                "total": total_sum,
                # total is the number of validate tokens
            }
        else:
            c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
            c_total = sum(log.get("c_total", 1) for log in logging_outputs)
            agg_output = {
                "uer": c_errors * 100.0 / c_total,
                "qua_error": e_len * 100.0 / c_total,
                "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
                "ntokens": ntokens,
                "nsentences": nsentences
            }

        return agg_output
