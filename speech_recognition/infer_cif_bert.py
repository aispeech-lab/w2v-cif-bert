#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import itertools as it
import editdistance
import logging
import os
import sys

import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data.data_utils import post_process
from fairseq.models.wav2vec.wav2vec2_cif import CIFFcModel
from examples.speech_recognition.cif_bert_decoder import CIF_BERT_Decoder


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_asr_eval_argument(parser):
    parser.add_argument("--lm_weight", type=float, default=0.2,
                        help="weight for lm while interpolating with neural score")
    parser.add_argument("--iscn", action='store_true', help="output char")
    parser.add_argument("--infer-threshold", type=float, default=None)

    return parser


def check_args(args):
    assert (
            args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
        args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id, trans
):
    for hypo in hypos[: min(len(hypos), 1)]:
        hyp_words = []

        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
            for hyp_word in hypo["words"]:
                if '[' not in hyp_word and '<' not in hyp_word:
                    continue
                hyp_words.append(hyp_word)
        else:
            hyp_pieces = tgt_dict.tokenizer.decode(hypo["tokens"].int().cpu())
            for hypo_chr in hyp_pieces.split():
                if '[' not in hypo_chr and '<' not in hypo_chr:
                    hyp_words.append(hypo_chr)
        tgt_words = []
        for tgt_word in trans.strip().split():
            if '[' not in tgt_word and '<' not in tgt_word:
                tgt_words.append(tgt_word)

        tgt_words = ' '.join(tgt_words)
        hyp_words = post_process(' '.join(hyp_words))

        if args.iscn:
            hyp_words = ' '.join(list(hyp_words.replace(' ', '')))
            tgt_words = ' '.join(list(tgt_words.replace(' ', '')))

        if res_files is not None:
            print("{} ({}-{})".format(hyp_words, speaker, id), file=res_files["hypo.words"])

        print("{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"])
        # only score top hypothesis
        if not args.quiet:
            logger.debug("HYPO:" + hyp_words)
            logger.debug("TARGET:" + tgt_words)
            logger.debug("___________________")

        hyp_words = hyp_words.split()
        tgt_words = tgt_words.split()
        return editdistance.eval(hyp_words, tgt_words), len(tgt_words)


def prepare_result_files(args):
    def get_res_file(file_prefix):
        if args.num_shards > 1:
            file_prefix = f'{args.shard_id}_{file_prefix}'
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    if not args.results_path:
        return None

    return {
        "hypo.words": get_res_file("hypo.word"),
        "ref.words": get_res_file("ref.word")
    }


def load_models_and_criterions(filenames, data_path, arg_overrides=None, task=None, model_state=None):
    models = []
    criterions = []

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides['wer_args'] = None
    arg_overrides['data'] = data_path

    if filenames is None:
        assert model_state is not None
        filenames = [0]
    else:
        filenames = filenames.split(":")

    for filename in filenames:
        if model_state is None:
            if not os.path.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
        else:
            state = model_state

        args = state["args"]
        if task is None:
            task = tasks.setup_task(args)
        model = task.build_model(args)
        model.load_state_dict(state["model"], strict=True)
        models.append(model)

        criterion = task.build_criterion(args)
        if "criterion" in state:
            criterion.load_state_dict(state["criterion"], strict=True)
        criterions.append(criterion)
    return models, criterions, args


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation
    """
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, prefix_tokens=None):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception('invalid sizes')
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)


def main(args, task=None, model_state=None):
    check_args(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 4000000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    if task is None:
        # Load dataset splits
        task = tasks.setup_task(args)
        task.load_dataset(args.gen_subset)
        logger.info(
            "| {} {} {} examples".format(
                args.data, args.gen_subset, len(task.dataset(args.gen_subset))
            )
        )

    all_trans = []
    if 'audio' in args.task:
        """
            tasks that load tsv data
            trans_path: raw trans (before bpe)
        """
        trans_path = os.path.join(args.data, "{}.word".format(args.gen_subset))
        with open(trans_path, "r") as f:
            for line in f:
                all_trans.append(line)

    # Set dictionary
    tgt_dict = task.target_dictionary

    logger.info("| decoding with criterion {}".format(args.criterion))

    # Load ensemble

    logger.info("| loading model(s) from {}".format(args.path))
    models, criterions, _ = load_models_and_criterions(
        args.path,
        data_path=args.data,
        arg_overrides=eval(args.model_overrides),  # noqa
        task=task,
        model_state=model_state,
    )
    optimize_models(args, use_cuda, models)

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task, models)

    # Initialize generator
    gen_timer = StopwatchMeter()

    generator = CIF_BERT_Decoder(args, task.target_dictionary)

    num_sentences = 0

    if args.results_path is not None and not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    res_files = prepare_result_files(args)
    errs_t = 0
    lengths_t = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample["target"][:, : args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample["id"].tolist()):
                speaker = None
                # id = task.dataset(args.gen_subset).ids[int(sample_id)]
                id = sample_id
                toks = sample["target"][i, :] if 'target_label' not in sample else sample["target_label"][i, :]
                target_tokens = (
                    utils.strip_pad(toks, tgt_dict.pad()).int().cpu()
                )
                trans = all_trans[id] if all_trans else task.dataset(args.gen_subset).ids[sample_id][1]['output']['text'].strip()
                # Process top predictions
                errs, length = process_predictions(
                    args, hypos[i], None, tgt_dict, target_tokens, res_files, speaker, id, trans
                )
                errs_t += errs
                lengths_t += length

            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    wer = None

    if lengths_t > 0:
        wer = errs_t * 100.0 / lengths_t
        logger.info(f"WER: {wer}")

    logger.info(
        "| Processed {} sentences ({} tokens) in {:.1f}s ({:.2f}"
        "sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
            )
    )
    logger.info("| Generate {} with beam={}".format(args.gen_subset, args.beam))

    return task, wer


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
