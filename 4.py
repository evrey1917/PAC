import numpy as np

def func(mas, window, a):
    if (window > len(mas)):
        return a
    a = np.append(a, sum(mas[:window]))
    return func(mas[1:], window, a)


# mas = np.array([2,4,3,5,2,3,1,1,2,2,3,2,2,9,4,8,3,8,2,4,4,5,6,4,6,8,9])
mas = np.array([2,3,4])

window = 4

b = func(mas, window, np.array([]))

print(b)