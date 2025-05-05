import torch
from normalizes import init_lecun_normal
from torch.nn import functional as F
from torch import nn, einsum
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


class SequenceWeight(nn.Module):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(self.dim)

        self.to_q = nn.Linear(d_msa, n_head * d_hidden)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden)
        self.dropout = nn.Dropout(p_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)

    def forward(self, msa):
        b, n, l = msa.shape[:3]
        tar_seq = msa[:, 0]
        # row
        q = self.to_q(tar_seq).view(b, 1, l, self.h, self.dim)
        k = self.to_k(msa).view(b, n, l, self.h, self.dim)
        q = q * self.scale
        attn = einsum('bqihd,bkihd->bkihq', q, k)
        attn = F.softmax(attn, dim=1)
        return self.dropout(attn)


class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)

        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)

        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias:
        self.to_b = init_lecun_normal(self.to_b)

        # gate:
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # out:
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair):
        b, n, l = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        #
        seq_weight = self.seq_weight(msa)  # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(b, n, l, self.h, self.dim)
        key = self.to_k(msa).reshape(b, n, l, self.h, self.dim)
        value = self.to_v(msa).reshape(b, n, l, self.h, self.dim)
        bias = self.to_b(pair)  # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        key = key * self.scaling
        attn = einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(b, n, l, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out


class MSAColAttention(nn.Module):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)

        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)

        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameters()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        b, n, l = msa.shape[:3]
        msa = self.norm_msa(msa)

        query = self.to_q(msa).reshape(b, n, l, self.h, self.dim)
        key = self.to_k(msa).reshape(b, n, l, self.h, self.dim)
        value = self.to_v(msa).reshape(b, n, l, self.h, self.dim)
        gate = torch.sigmoid(self.to_g(msa))

        query = query * self.scaling
        attn = einsum('bqihd,bkihd->bihqk', query, key)
        attn = F.softmax(attn, dim=-1)

        out = einsum('bihqk,bkihd->bqihd', attn, value).reshape(B, N, L, -1)
        out = gate * out

        out = self.to_out(out)
        return out


class BiasedAxialAttention(nn.Module):
    # Pair to Pair
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super().__init__()

        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_pair)

        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        self.to_b = init_lecun_normal(self.to_b)

        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias):
        b, l = pair.shape[:2]

        if self.is_row:
            pair = pair.permute(0, 2, 1, 3)
            bias = bias.permute(0, 2, 1, 3)

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)

        query = self.to_q(pair).reshape(b, l, l, self.h, self.dim)
        key = self.to_k(pair).reshape(b, l, l, self.h, self.dim)
        value = self.to_v(pair).reshape(b, l, l, self.h, self.dim)
        bias = self.to_b(bias)
        gate = torch.sigmoid(self.to_g(pair))

        query = query * self.scaling
        key = key / math.sqrt(l)
        attn = einsum('bnihk,bnjhk->bijh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)

        out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(b, l, l, -1)
        out = gate * out

        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0,2,1,3)
        return out

