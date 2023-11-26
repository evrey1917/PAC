from math import exp
from sklearn.metrics import log_loss

def sigmoid(x):
    return 1 / (1 + (exp(-x)))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self, in_neuron):
        # 0,5 просто потому что я так захотел (чтоб хоть что-то менялось, очевидно)
        self._speed = 0.01
        self._weigths = [0.5 for i in range(in_neuron)]
    
    def forward(self, x):
        # вектор x пришёл на вход, домножаем на веса
        x_forwarder = [x[i] * self._weigths[i] for i in range(len(x))]
        # суммируем и применяем функцию активации (сигмоида в данном случае, но можно использовать и другие)
        return sigmoid(sum(x_forwarder))
    
    def backward(self, x, loss):
        new_weigths = [self._weigths[i] + self._speed * deriv_sigmoid(x[i]) * x[i] for i in range(len(x))]
        self._weigths = new_weigths
        return self.forward(x)

class Model:
    def __init__(self):
        self.n1 = Neuron(3)
        self.n2 = Neuron(3)
        self.n3 = Neuron(3)

    def forward(self, x):
        if (len(x) != 2):
            raise ValueError("Input parametrs must be in array of length 2")
        y1 = self.n1.forward([1, x[0], x[1]])   # результат от первого нейрона
        y2 = self.n2.forward([x[0], x[1], 1])   # результат от второго нейрона
        y3 = self.n3.forward([1, y1, y2])       # результат от выходного (третьего) нейрона
        return y3
    
    def backward(self, x, loss):
        # delta_n3 = loss
        delta_n1 = self.n3._weigths[1] * loss
        delta_n2 = self.n3._weigths[2] * loss
        y1 = self.n1.backward([1, x[0], x[1]], delta_n1)
        y2 = self.n2.backward([x[0], x[1], 1], delta_n2)
        self.n3.backward([1, y1, y2], loss)

model = Model()
X = []
label = []

for i in range(len(X)):
    y = model.forward(X[i])
    err = log_loss(y, label[i])     # существует множество loss алгоритмов
    model.backward(X[i], err)