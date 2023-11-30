from math import exp
from random import random

def sigmoid(x):
    return 1 / (1 + (exp(-x)))

def deriv_sigmoid(x):
    return x * (1 - x)

def loss(x1, x2):
    return x1 - x2

class Neuron:
    def __init__(self, in_neuron):
        # 0,5 просто потому что я так захотел (чтоб хоть что-то менялось, очевидно)
        self._speed = 0.1
        self._in = 0
        self._activation = 0
        self._weigths = [random() for i in range(in_neuron)]
    
    def forward(self, x):
        # вектор x пришёл на вход, домножаем на веса
        self._activation = sigmoid(sum([x[i] * self._weigths[i] for i in range(len(x))]))
        # суммируем и применяем функцию активации (сигмоида в данном случае, но можно использовать и другие)
        return self._activation
    
    def backward(self, x, loss):
        # x - выходы из предыдущего слоя для градиента!!!
        delta = loss * deriv_sigmoid(self._activation)
        new_weights = [self._weigths[i] - self._speed * delta * x[i] for i in range(len(x))]
        self._weigths = new_weights

class Model:
    def __init__(self):
        self.n1 = Neuron(3)
        self.n2 = Neuron(3)
        self.n3 = Neuron(3)

    def forward(self, x):
        if (len(x) != 2):
            raise ValueError("Input parametrs must be in array of length 2")
        y1 = self.n1.forward([1, x[0], x[1]])      # результат от первого нейрона
        y2 = self.n2.forward([x[0], x[1], 1])      # результат от второго нейрона
        y3 = self.n3.forward([1, y1, y2])       # результат от выходного (третьего) нейрона
        return y3

    def backward(self, x, loss):
        if (len(x) != 2):
            raise ValueError("Input parametrs must be in array of length 2")
        
        delta1 = self.n3._weigths[1] * loss * deriv_sigmoid(self.n3._activation)
        delta2 = self.n3._weigths[2] * loss * deriv_sigmoid(self.n3._activation)
        self.n1.backward([1, x[0], x[1]], delta1)
        self.n2.backward([x[0], x[1], 1], delta2)
        self.n3.backward([1, self.n1._activation, self.n2._activation], loss)

model = Model()
X = [[0,0], [1,0], [0,1], [1,1]]
label = [0,1,1,0]

for k in range(40000):
    for i in range(len(X)):
        y = model.forward(X[i])
        err = loss(y, label[i])     # существует множество loss алгоритмов
        model.backward(X[i], err)

print("Results:")
print(model.forward(X[0]))
print(model.forward(X[1]))
print(model.forward(X[2]))
print(model.forward(X[3]))
