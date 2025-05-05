#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.nn.functional as F
import torch
from torch import nn


class RMSNorm(nn.Module):
    # Root Mean Square Layer Norm
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, self.dim, self.weight, self.eps)
