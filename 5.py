import math

N = int(input())

mas = [1] * (N + 1)
mas[0] = 0
mas[1] = 0

endus = math.ceil(math.sqrt(N)) + 1
for i in range(1, endus):
    if(mas[i] == 1):
        k = 2
        while k * i <= N:
            mas[k * i] = 0
            k += 1

if (mas[N] == 0):
    print("COMPLEX")
else: print("SIMPLE")

a = input()
