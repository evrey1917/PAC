import random

dig = random.randint(100,999)

print(dig)

summ = (dig % 10 + (dig // 10) % 10 + dig // 100)

print(summ)

z = input()
