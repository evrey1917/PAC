import torch
import torch.nn as nn
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        """Регистрация блоков"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=9)    # 3*19*19 -> 8*18*18
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=4)   # 8*9*9 -> 16*8*8
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)        # 8*18*18 -> 8*9*9 and 16*8*8 -> 16*4*4
        self.relu = nn.ReLU()                                               # activation

    def forward(self, x):
        """Прямой проход"""
        h = self.conv1(x)
        h = self.relu(h)
        h = self.pool(h)
        h = self.conv2(h)
        h = self.relu(h)
        y = self.pool(h)
        return y

x = torch.randn(4, 3, 19, 19)
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
#     if x.shape == (19, 19, 3):
#         return (x.permute(*torch.arange(x.ndim - 1, -1, -1))).view(3, 19, 19)
#     if x.shape == (3, 19, 19):
#         return x.view(3, 19, 19)
#     exit("Wrong matrice size")

# x = torch.randn(19, 19, 3)

# model = nn.Sequential(
#     EvroModel(preprocess),
#     nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=9),    # 3*19*19 -> 8*18*18
#     nn.ReLU(),                                              # activation
#     nn.MaxPool2d(kernel_size=5, stride=2, padding=2),       # 8*18*18 -> 8*9*9
#     nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=4),   # 8*9*9 -> 16*8*8
#     nn.ReLU(),                                              # activation
#     nn.MaxPool2d(kernel_size=5, stride=2, padding=2),       # 16*8*8 -> 16*4*4
# )

# print(model(x).shape)