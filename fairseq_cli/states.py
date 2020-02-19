#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Encode data from a file with a trained encoder-decoder model and save the model states.
"""

import logging
import os
import torch

from fairseq import options
from fairseq.models import BaseFairseqModel


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not (os.path.isdir(args.states_dir) and os.listdir(args.states_dir)), \
        '--output directory for encoder states must be empty'

    return _main(args)


def _main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger('fairseq_cli.states')

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load model
    logger.info('loading model from {}'.format(args.path))
    dirname, filename = os.path.split(args.path)
    model = BaseFairseqModel.from_pretrained(dirname, filename,
                                             tokenizer='moses',
                                             bpe='fastbpe',
                                             encoder_states_dir=args.states_dir)

    # create output directory if needed
    os.makedirs(args.states_dir, exist_ok=True)

    # reading sentences from file
    logger.info('encoding sentences from {}'.format(args.data))
    with open(args.data) as datafile:
        enc_sents = [sent.rstrip() for sent in datafile.readlines()]

    if use_cuda:
        model.cuda()

    _ = model.translate(enc_sents, states=True)

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

    return X




def cli_main():
    parser = options.get_states_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
