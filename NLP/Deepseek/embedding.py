#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
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


def precompute_freq_cis(args):
    dim = args.qk_rope_head_dim
    seq_len = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seq_len > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freq = freq / factor * (1 - smooth) + freq * smooth

    t = torch.arange(seq_len)
    freq = torch.outer(t, freq)
    freq_cis = torch.polar(torch.ones_like(freq), freq)
    return freq_cis


def find_correction_dim(num_rotations, dim, base, max_seq_len):
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_factor(min_i, max_i, dim):
    if min_i == max_i:
        max_i += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_i) / (max_i - min_i)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def apply_rotary_emb(x, freq_cis):
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freq_cis = freq_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freq_cis).flatten(3)
    return y.to(dtype)

