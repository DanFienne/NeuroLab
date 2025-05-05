#!/usr/bin/env python
# -*- coding:utf-8 -*-
from config import world_size, rank
from layers import Gate, Expert, MLP
import torch.distributed as dist
from config import ModelArgs
from torch import nn
import torch


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super(MoE, self).__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        # 选择当前节点，有哪些专家网络
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
             for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        # 统计专家层被x选中的次数 例如 [zj0, zj2, zj1, zj2, zj2, zj3] 中,
        # 假如是5个专家， counts=[1,1,3,1,0]，其中，专家4没人选，是 0
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            # idx, top 相当于专家找到在一个 batch 中，哪些 x 属于他，并与对应的 gate 的 weight 计算，相当于放缩参数
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # 共享专家，全局共享的专家
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)
