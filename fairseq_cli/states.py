#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Encode pre-processed data with a trained encoder-decoder model and save the model states.
"""

import logging
import math
import os
import sys

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'
    assert not (os.path.isdir(args.states_dir) and os.listdir(args.states_dir)), \
        '--output directory for encoder states must be empty'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.states')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # create output directory if needed
    os.makedirs(args.states_dir, exist_ok=True)

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
        model.change_encoder_states_dir(args.states_dir)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    generator = task.build_generator(args)

    logger.info('performing forward pass and saving encoder states to {}'.format(args.states_dir))
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            _ = task.inference_step(generator, models, sample, prefix_tokens)

    # reading the tensor files to a list of tensors
    logger.info('loading encoder states from {}'.format(args.states_dir))
    _, _, filenames = next(os.walk(args.states_dir))
    tensors = [torch.load(os.path.join(args.states_dir, filename)) for filename in filenames]

    # tensors have shape of time_steps x batch_size x length
    if args.states_operation == 'pad':
        most_time_steps = max(tensors, key=lambda x: x.shape[0]).shape[0]
        logger.info('creating output tensor with shape [n_sents, length * max(time_steps)]')

        # reshaping individual tensors
        for i, tensor in enumerate(tensors):
            target = torch.zeros(most_time_steps, tensor.shape[1], tensor.shape[2])
            target[:tensor.shape[0], :, :] = tensor # padding with zeros to max(time_steps)
            # creating a tensor with one row per sentence
            tensors[i] = target.transpose(0, 1).reshape(target.shape[1], -1)

        # concatenating tensors and writing to file
        X = torch.cat(tensors, dim=0)
        torch.save(X, os.path.join(args.states_dir, 'X_padded.pt'))
        logger.info('model states feature vector saved to {}'.format(os.path.join(args.states_dir, 'X_padded.pt')))

    elif args.states_operation == 'average':
        logger.info('creating output tensor with shape [n_sents, length]')
        # averaging all relevant (no padding) time steps
        tensors = [tensor.mean(0) for tensor in tensors]
        X = torch.cat(tensors, dim=0)
        torch.save(X, os.path.join(args.states_dir, 'X_averaged.pt'))
        logger.info('model states feature vector saved to {}'.format(os.path.join(args.states_dir, 'X_averaged.pt')))




def cli_main():
    parser = options.get_states_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
