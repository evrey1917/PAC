def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

n = int(input())

for i in range(0, n):
    print(fib(i))
