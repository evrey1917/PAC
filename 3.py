import numpy as np

mas = np.array(([[2,4,3,5,2,3], [3,3,0,5,1,4], [2,5,7,3,2,2]], [[2,4,3,5,2,3], [7,3,0,5,7,4], [2,5,7,3,2,2]], [[2,4,3,5,2,3], [7,3,0,5,7,4], [2,5,7,3,2,2]]), dtype=np.uint8)

new_mas = []
for i in range(len(mas[0])):
    for k in range(len(mas[0][0])):
        new_mas.append([mas[0][i][k] + 1000*mas[1][i][k] + 1000*1000*mas[2][i][k]])

uniq = len(np.unique(new_mas))
print(uniq)