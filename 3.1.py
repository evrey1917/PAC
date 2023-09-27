import numpy as np

def magic(x, y, z):
    return x + y*1000 + z*1000000


mas = np.array(([[2,4,3,5,2,3], [3,3,0,5,1,4], [2,5,7,3,2,2]], [[2,4,3,5,2,3], [7,3,0,5,7,4], [2,5,7,3,2,2]], [[2,4,3,5,2,3], [7,3,0,5,7,4], [2,5,7,3,2,2]]), dtype=np.uint8)

mas0 = mas[0]
mas1 = mas[1]
mas2 = mas[2]

doMagic = np.vectorize(magic)
uniq = np.unique(doMagic(mas0, mas1, mas2))

print(len(uniq))