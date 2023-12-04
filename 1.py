import torch
import torch.nn as nn
import numpy as np

class EvroModel(nn.Module):
    def __init__(self, func):
        """Регистрация блоков"""
        super().__init__()
        self.func = func
        
    def forward(self, x):
        """Прямой проход"""
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 784)

a = 784
b = 10
x = torch.randn(10, a)
model = nn.Sequential(
    EvroModel(preprocess),
    nn.Linear(a, 32),
    nn.ReLU(),
    nn.Linear(32, b, bias=False),
    nn.ReLU()
)
print(model(x).shape)