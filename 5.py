import numpy as np

def func(mas, i, a):
    if (i >= len(mas[0])):
        return a.reshape(int(len(a) / 3), 3)
    
    if (mas[0][i] < mas[1][i] + mas[2][i]
        and mas[1][i] < mas[0][i] + mas[2][i]
        and mas[2][i] < mas[0][i] + mas[1][i]):

        a = np.append(a, [mas[0][i], mas[1][i], mas[2][i]])
        
    return func(mas, i + 1, a)


mas = np.array(([[2,4,3,5,2,3], [3,3,0,5,1,4], [2,5,7,3,2,2]]), dtype=np.uint8)

b = func(mas, 0, np.array([[]]))
print(b)