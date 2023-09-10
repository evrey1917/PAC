import random

a = 0
b = 100

n = random.randint(a, b)
rand = [random.randint(a, b) for i in range(n)]

##print(rand)

lenght = len(rand)
count = 0

for i in rand:
    if (i % 2 == 0):
        count += 1

print("even: {0}, odd: {1}".format(count, lenght - count))
