#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class ParallelEmbedding(nn.Module):
    # embedding layer with parallelism support across distributed process
    def __init__(self, vocab_size, dim, rank, world_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.world_size = world_size
        # split vocab by world_size
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 相当于对 embedding 的 weight 做了 row-parallel, 后面要调用 all reduce 补回
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x):
        mask = 0
        if self.world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if self.world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y
