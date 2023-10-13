import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
from os import listdir

class Nails:
    index = 0
    len = 0
    photo_or_mask = 0
    pairs = []

    def init(self, pairs):
        self.len = pairs.shape[1]
        self.pairs = pairs
        self.show(self.pairs[0])

    def next(self, event):
        self.index = self.index + 1
        if (self.index >= self.len):
            self.index = 0
        self.show(self.pairs[self.index])
    
    def prev(self, event):
        self.index = self.index - 1
        if (self.index < 0):
            self.index = self.len - 1
        self.show(self.pairs[self.index])

    def show(self, file):
        axes.clear()
        _, thresh = cv2.threshold(cv2.cvtColor(cv2.imread(file[1]), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        axes.imshow(cv2.drawContours(cv2.imread(file[0]), contours, -1, (0,255,0), 3)[:, :, ::-1])
        plt.show()

list1 = np.array(listdir('./labels'))

new_pairs = pd.DataFrame([*map(lambda x: './images/' + x, list1)])
new_pairs[1] = [*map(lambda x: './labels/' + x, list1)]
new_pairs = new_pairs.T


window, axes = plt.subplots()
window.subplots_adjust(bottom = 0.2)


on_screen = Nails()

axnext = window.add_axes([0.8, 0.1, 0.05, 0.075])
axprev = window.add_axes([0.75, 0.1, 0.05, 0.075])


next = Button(axnext, 'Next', color = 'lightblue', hovercolor = 'blue')
next.on_clicked(on_screen.next)

prev = Button(axprev, 'Prev', color = 'lightblue', hovercolor = 'blue')
prev.on_clicked(on_screen.prev)


on_screen.init(new_pairs)