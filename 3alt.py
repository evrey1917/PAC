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
    _, thresh = cv2.threshold(cv2.cvtColor(cv2.imread(new_pairs[1][i]), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('okno', cv2.drawContours(cv2.imread(new_pairs[0][i]), contours, -1, (0,255,0), 3))