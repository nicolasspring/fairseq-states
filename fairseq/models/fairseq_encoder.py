# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch.nn as nn
from torch import save
from fairseq.modules import exp_path_search


class FairseqEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self._encoder_states_dir = None

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def change_encoder_states_dir(self, encoder_states_dir):
        """Sets the output directory for saving encoder model states."""
        self._encoder_states_dir = encoder_states_dir

    def _save_encoder_state(self, state, pattern):
        """Saves the encoder state to a file with a specified *pattern*."""
        file_pattern = os.path.join(self._encoder_states_dir, pattern)
        outfile = exp_path_search(file_pattern)
        save(state, outfile)
