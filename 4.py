import numpy as np
import cv2

vidos = cv2.VideoCapture('damn.mp4')

while 1:
    check, frame = vidos.read()
    if (not(check)):
        break
    cv2.imshow('vidos', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cv2.waitKey(1000//60)