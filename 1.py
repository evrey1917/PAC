import pandas as pd
import numpy as np

data = np.random.random(size = (10, 5))

frame = pd.DataFrame(data)

for i in range(frame.iloc[0].count() - 1):
    mas = frame.iloc[i].loc[lambda x: x > 0.3]
    print("Medium of ", i, " stroka: ", mas.sum() / mas.count())

# for i in range(0, 10):
#     n = 0
#     sum = 0
#     for j in range(0, 5):
#         if frame[j][i] > 0.3:
#             sum += frame[j][i]
#             n += 1
#     print("Medium of ", i, " stroka: ", sum/n)