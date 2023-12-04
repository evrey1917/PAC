import torch
import torch.nn as nn
import numpy as np
from math import exp

def relu(input):
    size = input.shape
    a = input.flatten().tolist()
    b = list(map(lambda x: max(0,x), a))
    answer = torch.tensor(b).view(size)
    return answer

def tanh(input):
    size = input.shape
    a = input.flatten().tolist()
    b = list(map(lambda x: (exp(x) - exp(-x))/(exp(x) + exp(-x)), a))
    answer = torch.tensor(b).view(size)
    return answer

def soft(input):
    size = input.shape
    a = input.flatten().tolist()
    b = list(map(lambda x: exp(x), a))
    summa = sum(b)
    c = list(map(lambda x: x/summa, b))
    answer = torch.tensor(c).view(size)
    return answer

class EvroModel(nn.Module):
    def __init__(self):
        """Регистрация блоков"""
        super().__init__()
        # weights
        self.w1 = torch.randn(256, 64)
        self.w2 = torch.randn(64, 16)
        self.w3 = torch.randn(16, 4)
        # biases
        self.b1 = torch.randn(1, 64)
        self.b2 = torch.randn(1, 16)
        self.b3 = torch.randn(1, 4)
        
    def forward(self, x):
        """Прямой проход"""
        h = x @ self.w1 + self.b1
        h = relu(h)
        h = h @ self.w2 + self.b2
        h = tanh(h)
        h = h @ self.w3 + self.b3
        y = soft(h)
        return y

x = torch.randn(1, 256)
a = EvroModel()
print(a.forward(x))
# print(k.shape)