import cv2
import numpy as np

img1 = cv2.imread('candy_ghost.png')
img2 = cv2.imread('lab7.png')

g_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(g_img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(g_img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

cv2.imwrite("as.png", img3)