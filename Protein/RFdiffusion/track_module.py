from torch import nn
import torch
import torch.nn.functional as F

from normalizes import init_lecun_normal
from attentions import MSARowAttentionWithBias, MSAColAttention, BiasedAxialAttention
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


class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super().__init__()
        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_rbf.bias)

        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat):
        b, l = pair.shape[:2]
        rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))

        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat))
        pair = pair + self.ff(pair)
        return pair

