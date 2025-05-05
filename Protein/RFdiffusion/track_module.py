from torch import nn
import torch

from normalizes import init_lecun_normal
from attentions import MSARowAttentionWithBias, MSAColAttention
from layers import Dropout, FeedForwardLayer


class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super().__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair + 36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden)
        self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden)
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        self.reset_parameters()

    def reset_parameters(self):
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state):
        b, n, l = msa.shape[:3]

        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)

        state = self.norm_state(state)
        state = self.proj_state(state).reshape(b, 1, l, -1)
        msa = msa.index_add(1, torch.tensor([0], device=state.device), state)

        # apply row/col attention to msa & transform
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)
        return msa

