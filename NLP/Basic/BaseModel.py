#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch import nn
import torch


class BaseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layer(x)


def data_loaders(batch_size, train_data, test_data, train_label, test_label):
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.long)

    train_data = TensorDataset(train_data, train_label)
    test_data = TensorDataset(test_data, test_label)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train(device, batch_size, train_data, test_data, train_label, test_label):
    train_loader, test_loader = data_loaders(batch_size, train_data, test_data, train_label, test_label)
    model = BaseModel(input_size=train_data.size()[1], output_size=train_label.size()[1]).to(device)
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            eval_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=1)
            correct += torch.sum(y_pred == y).item()
            total += y.size(0)
    acc = correct / total
    return train_loss, eval_loss, acc
