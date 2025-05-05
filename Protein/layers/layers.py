import torch
from torch import nn


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

