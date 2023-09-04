import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('integ', type=int)

A = parser.parse_args()

N = A.integ
mas = [0] * N

for i in range (0, N):
    mas[i] = random.random()

print(mas)

check = 1
while check == 1:
    check = 0
    for i in range(0, N - 1):
        if (mas[i] > mas[i + 1]):
            buf = mas[i]
            mas[i] = mas[i + 1]
            mas[i + 1] = buf
            check = 1

print(mas)

a = input()
