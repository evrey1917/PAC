import torch
import torch.nn as nn
import numpy as np

class FirstModel(nn.Module):
    def __init__(self):
        """Регистрация блоков"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=9)    # 3*19*19 -> 8*18*18
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=4)   # 8*9*9 -> 16*8*8
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)        # 8*18*18 -> 8*9*9 and 16*8*8 -> 16*4*4
        self.relu = nn.ReLU()                                               # activation

    def forward(self, x):
        """Прямой проход"""
        flag = 0
        if x.shape == (19, 19, 3):
            x = x.permute(*torch.arange(x.ndim - 1, -1, -1))
            flag = 1

        if x.shape == (3, 19, 19):
            h = self.conv1(x)
            h = self.relu(h)
            h = self.pool(h)
            h = self.conv2(h)
            h = self.relu(h)
            y = self.pool(h)

            # Возвращаем, в каком формате и дали
            if flag:
                return y.permute(*torch.arange(x.ndim - 1, -1, -1))
            else:
                return y
        else:
            exit("Wrong matrice size")

class SecondModel(nn.Module):
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

class FinalModel(nn.Module):
    def __init__(self):
        """Регистрация блоков"""
        super().__init__()
        self.first_model = FirstModel()
        self.second_model = SecondModel()
        
    def forward(self, x):
        """Прямой проход"""
        h = self.first_model(x)
        y = self.second_model(h.reshape((1, 256)))
        return y.reshape([-1])

x = torch.randn(3, 19, 19)
model = FinalModel()

result = model(x)
print(result)
print(result.shape)