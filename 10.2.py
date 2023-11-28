from math import exp

def sigmoid(x):
    return 1 / (1 + (exp(-x)))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def loss(x1, x2):
    return x1 - x2

# class Neuron:
#     def __init__(self, in_neuron):
#         # 0,5 просто потому что я так захотел (чтоб хоть что-то менялось, очевидно)
#         self._speed = 0.3
#         self._in = 0
#         self._weigths = [0.5 for i in range(in_neuron)]
    
#     def forward(self, x):
#         # вектор x пришёл на вход, домножаем на веса
#         self._in = sum([x[i] * self._weigths[i] for i in range(len(x))])
#         # суммируем и применяем функцию активации (сигмоида в данном случае, но можно использовать и другие)
#         return sigmoid(self._in)
    
#     def backward(self, x, loss):
#         grad = deriv_sigmoid(self._in) * loss
#         delta = deriv_sigmoid(self._in) * loss
#         new_weigths = [self._weigths[i] - self._speed * x[i] * delta for i in range(len(x))]
#         self._weigths = new_weigths
#         return delta, self.forward(x)

class Neuron:
    def __init__(self, in_neuron):
        # 0,5 просто потому что я так захотел (чтоб хоть что-то менялось, очевидно)
        self._speed = 0.5
        self._in = 0
        self._activation = 0
        self._weigths = [0.5 for i in range(in_neuron)]
    
    def forward(self, x):
        # вектор x пришёл на вход, домножаем на веса
        # print([x[i] * self._weigths[i] for i in range(len(x))])
        self._in = sum([x[i] * self._weigths[i] for i in range(len(x))])
        self._activation = sigmoid(self._in)
        # суммируем и применяем функцию активации (сигмоида в данном случае, но можно использовать и другие)
        return self._activation
    
    def backward(self, x, loss):
        # x - выходы из предыдущего слоя для градиента!!!
        delta = loss * deriv_sigmoid(self._in)
        # deltas = [self._weigths[i] * delta for i in range(len(x))]
        new_weights = [self._weigths[i] - self._speed * delta * x[i] for i in range(len(x))]
        self._weigths = new_weights
        # дельты для слоя ниже
        # return deltas

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
        if (len(x) != 2):
            raise ValueError("Input parametrs must be in array of length 2")
        # delta_n3 = loss
        # delta_n1 = self.n3._weigths[1] * loss
        # delta_n2 = self.n3._weigths[2] * loss

        delta1 = self.n3._weigths[1] * loss
        delta2 = self.n3._weigths[2] * loss
        self.n1.backward([1, x[0], x[1]], delta1)
        self.n2.backward([x[0], x[1], 1], delta2)
        self.n3.backward([1, self.n1.forward([1, x[0], x[1]]), self.n2.forward([x[0], x[1], 1])], loss)
        # deltas = self.n3.backward([1, self.n1._activation, self.n2._activation], loss)
        # self.n1.backward([1, x[0], x[1]], deltas[1])
        # self.n2.backward([x[0], x[1], 1], deltas[2])

        # delta_n3 = loss * deriv_sigmoid(self.n3._in)
        # delta1, y1 = self.n1.backward([1, x[0], x[1]], delta_n3)
        # delta2, y2 = self.n2.backward([x[0], x[1], 1], delta_n3)
        # self.n3.backward([1, y1, y2], loss)

model = Model()
X = [[0,0], [1,0], [0,1], [1,1]]
label = [0,1,1,0]

# for k in range(50000):
for i in range(len(X)):
    y = model.forward(X[i])
    err = loss(y, label[i])     # существует множество loss алгоритмов
    print(err)
    model.backward(X[i], err)

# print("Results:")
# print(model.forward(X[0]))
# print(model.forward(X[1]))
# print(model.forward(X[2]))
# print(model.forward(X[3]))
