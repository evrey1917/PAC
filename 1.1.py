import numpy as np

mas = np.array([2,4,3,5,2,3,1,1,2,2,3,2,2,9,4,8,3,8,2,4,4,5,6,4,6,8,9])

mas, freq = np.unique(mas, return_counts=True)

sort_ind = np.argsort(freq)[::-1]
vals = mas[sort_ind]
freq = freq[sort_ind]
mas = np.repeat(vals, freq)

print(sort_ind, mas, freq, vals)