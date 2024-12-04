"""
Modified from https://github.com/karpathy/makemore/blob/master/makemore.py
- Added comments with shapes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Bigram language model


class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))  # Shape (27,27)

    def get_block_size(self):
        return 1  # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

        # 'forward pass', lol
        # idx shape: (N,L), e.g. (32,16); N = batch size, L = block size (max seq len)
        # targets shape: (N,L) as well
        logits = self.logits[idx]  # (N,L) -> (N,L,D), e.g. (32,16) -> (32,16,27)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                # Logits: (N,L,D) -> (N*L,D), e.g. (32,16,27) -> (512,27)
                logits.view(-1, logits.size(-1)),
                # Targets: (N,L) -> (N*L,), e.g. (32,16) -> (512,)
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss
