import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
from os import listdir

list1 = np.array(listdir('./labels'))

new_pairs = pd.DataFrame([*map(lambda x: './images/' + x, list1)])
new_pairs[1] = [*map(lambda x: './labels/' + x, list1)]

print(new_pairs)
print(new_pairs.shape[0])


window, axes = plt.subplots()
window.subplots_adjust(bottom = 0.2)


plt.show()