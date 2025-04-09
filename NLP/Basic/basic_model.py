#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 快速了解神经网络工作流程
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 初始化参数
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        # Linear
        out = x @ self.weight.T + self.bias
        return out


def train(lr=0.1, num_epochs=1000):
    inputs = torch.tensor(torch.randn(3, 3), dtype=torch.float32)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    model = Model()

    # 分类
    loss_fn = nn.CrossEntropyLoss()
    # 回归
    # loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        # 前向传播
        output = model(inputs)
        loss = loss_fn(output, labels)
        # 反向传播
        loss.backward()

        # 优化器: 手动更新 weight 和 bias
        with torch.no_grad():
            model.weight -= lr * model.weight.grad
            model.bias -= lr * model.bias.grad
        model.weight.grad.zero_()
        model.bias.grad.zero_()

        # 分类
        pred = torch.argmax(output, dim=1)
        print(pred)


if __name__ == '__main__':
    train()
