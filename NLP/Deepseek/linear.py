#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn.functional as F
from config import gemm_impl, block_size, world_size
from kernel import weight_dequant, act_quant, fp8_gemm
import torch.distributed as dist
from torch import nn
import torch


def linear(x, w, b=None):
    # y = xw^T + b.
    if w.element_size() > 1:
        return F.linear(x, w, b)
    elif gemm_impl == "bf16":
        w = weight_dequant(w, w.scale)
        return F.linear(x, w, b)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, w, w.scale)
        if b is not None:
            y += b
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter('scale', None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x):
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x):
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        return y



