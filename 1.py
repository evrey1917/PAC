import numpy as np

mas = np.array([2,4,3,5,2,3,1,1,2,2,3,2,2,9,4,8,3,8,2,4,4,5,6,4,6,8,9])

mas, freq = np.unique(mas, return_counts=True)

print(mas, freq)

new_mas = []
for i in range(len(freq)):
    max = 0
    for j in range(len(freq) - i):
        if freq[j] > freq[max]:
            max = j
    new_mas.append(mas[max])

    for j in range(max, len(freq) - i - 1):
        mas[j] = mas[ j + 1]
        freq[j] = freq[ j + 1]

mas = new_mas

print(mas)