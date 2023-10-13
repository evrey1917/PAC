import pandas as pd
import numpy as np
import cv2
from os import listdir

list1 = np.array(listdir('./labels'))

new_pairs = pd.DataFrame([*map(lambda x: './images/' + x, list1)])
new_pairs[1] = [*map(lambda x: './labels/' + x, list1)]

print(new_pairs)
print(new_pairs.shape[0])

i = 0
while i < new_pairs.shape[0] - 1:
    cv2.waitKey(0)
    i = i + 1
    cv2.imshow('okno', cv2.hconcat([cv2.imread(new_pairs[0][i]), cv2.imread(new_pairs[1][i])]))