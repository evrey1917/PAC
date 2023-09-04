import random

inf = 100000

dig = random.randint(0, inf)

summ = 0

print(dig)

while (dig > 0):
    summ += dig % 10
    dig = dig // 10

print(summ)

a = input()
