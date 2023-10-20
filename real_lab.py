import argparse
import cv2
import numpy as np
import sys
import random

sift = cv2.SIFT_create()

main = cv2.imread(sys.argv[1])

for src_file in sys.argv[2::]:
    g_main = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
    key_main, des_main = sift.detectAndCompute(g_main, None)

    img = cv2.imread(src_file)
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key_img, des_img = sift.detectAndCompute(g_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_img, des_main, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    
    src_pts = np.float32([key_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_main[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w, c = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    my_color = [random.randint(0, 255) for i in range (3)]
    img = cv2.polylines(img, [np.int32(dst)], True, my_color, 3, cv2.LINE_AA)
    main = cv2.polylines(main, [np.int32(dst)], True, my_color, 3, cv2.LINE_AA)

    draw_params = dict(matchColor = [random.randint(0, 255) for i in range (3)],
        singlePointColor = None,
        matchesMask = matchesMask,
        flags = 2)
    main = cv2.drawMatches(img, key_img, main, key_main, good, None, **draw_params)

cv2.imwrite("answer.png", main)


# python3 real_lab.py lab7.png candy_ghost.png pampkin_ghost.png