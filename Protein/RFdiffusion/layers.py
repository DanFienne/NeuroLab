import torch
from torch import nn
from torch.nn import functional as F


class Dropout(nn.Module):

    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super().__init__()
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1 - p_drop]))
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop
    def forward(self, x):
        if not self.training:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = -1
        mask = self.sampler.sample(shape).to(x.device).view(shape)
        x = x * mask / (1 - self.p_drop)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model * r_ff, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return x


