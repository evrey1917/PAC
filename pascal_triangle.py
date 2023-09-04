import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('integ', type=int)

A = parser.parse_args()

N = A.integ

mas = [[]]*N

for i in range(0, N):
    mas[i] = [1] * (i + 1)

for i in range(0, N):
    for j in range(1, i):
        mas[i][j] = mas[i - 1][j - 1] + mas[i - 1][j]

summ = N - 1

for i in mas[N - 1]:
    summ += len(str(i))

for i in range(0, N):
    summ_now = i + 1
    for k in mas[i]:
        summ_now += len(str(k))
    print(" " * math.ceil((summ - summ_now)/2),end = '')

    for j in range(0, i + 1):
        print(mas[i][j],end = ' ')
    print("")
