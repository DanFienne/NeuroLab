#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn.functional as F
from config import gemm_impl, block_size
from kernel import weight_dequant, act_quant, fp8_gemm
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

