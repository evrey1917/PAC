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
        self.show(self.pairs[0][0])

    def next(self, event):
        self.index = self.index + 1
        if (self.index >= self.len):
            self.index = 0
        self.show(self.pairs[self.index][self.photo_or_mask])
    
    def prev(self, event):
        self.index = self.index - 1
        if (self.index < 0):
            self.index = self.len - 1
        self.show(self.pairs[self.index][self.photo_or_mask])

    def change_photo_mask(self, event):
        self.photo_or_mask = abs(self.photo_or_mask - 1)
        self.show(self.pairs[self.index][self.photo_or_mask])

    def show(self, file):
        axes.clear()
        axes.imshow(cv2.imread(file)[:, :, ::-1])
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
axchange = window.add_axes([0.75, 0.02, 0.1, 0.075])


next = Button(axnext, 'Next', color = 'lightblue', hovercolor = 'blue')
next.on_clicked(on_screen.next)

prev = Button(axprev, 'Prev', color = 'lightblue', hovercolor = 'blue')
prev.on_clicked(on_screen.prev)

change = Button(axchange, 'Photo/mask', color = 'lightblue', hovercolor = 'blue')
change.on_clicked(on_screen.change_photo_mask)


on_screen.init(new_pairs)