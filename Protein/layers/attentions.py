import torch
from normalizes import init_lecun_normal
from torch.nn import functional as F
from torch import nn
import math


class AttentionWithBias(nn.Module):
    def __init__(self, dim_gate=256, dim_bias=128, num_heads=8, dim_hidden=32):
        super().__init__()
        # norm
        self.norm_gate = nn.LayerNorm(dim_gate)
        self.norm_bias = nn.LayerNorm(dim_bias)
        # linear
        self.to_q = nn.Linear(dim_gate, num_heads * dim_hidden, bias=False)
        self.to_k = nn.Linear(dim_gate, num_heads * dim_hidden, bias=False)
        self.to_v = nn.Linear(dim_gate, num_heads * dim_hidden, bias=False)
        self.to_b = nn.Linear(dim_bias, num_heads, bias=False)
        self.to_g = nn.Linear(dim_gate, num_heads * dim_hidden)
        self.to_out = nn.Linear(num_heads * dim_hidden, dim_gate)
        # param
        self.scaling = 1 / math.sqrt(dim_hidden)
        self.h = num_heads
        self.dim = dim_hidden
        # init param
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        # bias
        self.to_b = init_lecun_normal(self.to_b)
        # gate
        nn.init.zeros_(self.to_g.weight)
        nn.init.zeros_(self.to_g.bias)
        # out
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, bias):
        b, l = x.shape[:2]
        # normal
        x = self.norm_gate(x)
        bias = self.norm_bias(bias)
        # forward
        query = self.to_q(x).reshape(b, l, self.h, self.dim)
        key = self.to_k(x).reshape(b, l, self.h, self.dim)
        value = self.to_v(x).reshape(b, l, self.h, self.dim)
        bias = self.to_b(bias)
        gate = torch.sigmoid(self.to_g(x))

        key = key * self.scaling
        attn = torch.einsum('bqhd,bkhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)

        out = torch.einsum('bqkh,bkhd->bqhd', attn, value).reshape(b, l, -1)
        out = gate * out
        out = self.to_out(out)
        return out




