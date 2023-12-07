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

def relu_deriv(input):
    size = input.shape
    a = input.flatten().tolist()
    b = list(map(lambda x: 1 if x >= 0 else 0, a))
    answer = torch.tensor(b).view(size)
    return answer

def tanh(input):
    size = input.shape
    a = input.flatten().tolist()
    b = list(map(lambda x: (exp(x) - exp(-x))/(exp(x) + exp(-x)), a))
    answer = torch.tensor(b).view(size)
    return answer

def tanh_deriv(input):
    size = input.shape
    a = input.flatten().tolist()
    b = list(map(lambda x: 4/((exp(x) + exp(-x))**2), a))
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

def soft_deriv(input):
    answer = soft(input)
    answer = answer * (1 - answer)
    return answer

class EvroModel(nn.Module):
    def __init__(self):
        """Регистрация блоков"""
        super().__init__()
        # speed of renowa
        self.speed = 0.1
        # weights zeros
        self.wz1 = torch.zeros(256, 64, requires_grad=True)
        self.wz2 = torch.zeros(64, 16, requires_grad=True)
        self.wz3 = torch.zeros(16, 4, requires_grad=True)
        # biases
        self.b1 = torch.randn(1, 64, requires_grad=True)
        self.b2 = torch.randn(1, 16, requires_grad=True)
        self.b3 = torch.randn(1, 4, requires_grad=True)

    def forward(self, x):
        """Прямой проход"""
        h = x @ self.wz1 + self.b1
        h = relu(h)
        h = h @ self.wz2 + self.b2
        h = tanh(h)
        h = h @ self.wz3 + self.b3
        y = soft(h)
        return y

    def backward(self, x, y):
        out1 = x @ self.wz1 + self.b1
        act1 = relu(out1)
        out2 = act1 @ self.wz2 + self.b2
        act2 = tanh(out2)
        out3 = act2 @ self.wz3 + self.b3
        out_act3 = soft(out3)

        loss = torch.mean((out_act3 - y)**2)

        delta3 = loss * soft_deriv(out_act3)
        self.wz3 = self.wz3 - (act2.T @ delta3) * self.speed
        self.b3 = self.b3 - delta3 * self.speed
        
        delta2 = (delta3 @ self.wz3.T) * tanh_deriv(out2)
        self.wz2 = self.wz2 - (act1.T @ delta2) * self.speed
        self.b2 = self.b2 - delta2 * self.speed

        delta1 = (delta2 @ self.wz2.T) * tanh_deriv(out1)
        self.wz1 = self.wz1 - (x.T @ delta1) * self.speed
        self.b1 = self.b1 - delta1 * self.speed

a = EvroModel()

x = torch.randn(1, 256)
y = torch.randn(1, 4)

a.backward(x, y)
