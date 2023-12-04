import torch
import torch.nn as nn
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        """Регистрация блоков"""
        super().__init__()
        self.lin1 = nn.Linear(256, 64)  # Полносвязный слой 1
        self.lin2 = nn.Linear(64, 16)   # Полносвязный слой 2
        self.lin3 = nn.Linear(16, 4)    # Полносвязный слой 3
        self.relu = nn.ReLU()           # Функция активации 1
        self.tanh = nn.Tanh()           # Функция активации 2
        self.soft = nn.Softmax(dim=1)   # Функция активации 3
        
    def forward(self, x):
        """Прямой проход"""
        h = self.lin1(x)
        h = self.relu(h)
        h = self.lin2(h)
        h = self.tanh(h)
        h = self.lin3(h)
        y = self.soft(h)
        return y

x = torch.randn(9, 256)
model = SimpleModel()

result = model(x)
print(result)
print(result.shape)


# import torch
# import torch.nn as nn
# import numpy as np

# class EvroModel(nn.Module):
#     def __init__(self, func):
#         """Регистрация блоков"""
#         super().__init__()
#         self.func = func
        
#     def forward(self, x):
#         """Прямой проход"""
#         return self.func(x)

# def preprocess(x):
#     return x.view(10, 256)

# # 10 - количество векторов длины 256
# x = torch.randn(10, 256)
# model = nn.Sequential(
#     EvroModel(preprocess),
#     nn.Linear(256, 64),
#     nn.ReLU(),
#     nn.Linear(64, 16),
#     nn.Tanh(),
#     nn.Linear(16, 4),
#     nn.Softmax(dim=1)
# )

# print(model(x))