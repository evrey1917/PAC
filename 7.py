def gem_gen(fir, q):
    s = fir
    while (1):
        s *= q
        yield s

fit = gem_gen(1, 2)
for i in range(20):
    print(next(fit))
