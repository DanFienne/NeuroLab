#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.distributed as dist

from normalize import RMSNorm
from layers import Block
from linear import linear, Linear, ColumnParallelLinear
from embedding import ParallelEmbedding, precompute_freq_cis
import torch
from torch import nn
from config import ModelArgs, world_size, rank


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Transformer, self).__init__()

        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim, rank, world_size)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freq_cis(args), persistent=False)

    def forward(self, tokens, start_pos):
        seq_len = tokens.size(1)
        h = self.embed(tokens)
        freq_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freq_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

