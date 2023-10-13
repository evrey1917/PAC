import numpy as np
import cv2

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

timer_motion = 600

sigmas = 100000
blur_size = 91

# vidos = cv2.VideoCapture('a.mp4')

now_time = 0

vidos = cv2.VideoCapture(0, cv2.CAP_DSHOW)

secs = 1000//(int(vidos.get(cv2.CAP_PROP_FPS)) + 30)

height = 300
width = 300

red = np.zeros((height, width, 3), np.uint8)
green = np.zeros((height, width, 3), np.uint8)

red[:,:] = (0, 0, 255)
green[:,:] = (0, 255, 0)

check, frame_prev = vidos.read()
frame_prev = to_gray(frame_prev)

if (check):
    while 1:
        check, frame_now = vidos.read()
        if (not(check)):
            break
        frame_delta = cv2.absdiff(to_gray(frame_now), frame_prev)

        _, thresh = cv2.threshold(frame_delta, 60, 255, cv2.THRESH_BINARY)
        frame_show = frame_now.copy()

        contours, _ = cv2.findContours(cv2.GaussianBlur(thresh, (blur_size, blur_size), sigmas, None, sigmas), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(frame_show, (x,y), (x+w,y+h), (0, 0 ,255), 2)

        if contours == () and now_time == 0:
            cv2.imshow('motion', green)
            cv2.putText(frame_show, "OK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            if contours != ():
                now_time = timer_motion
            else:
                now_time = now_time - secs
                if (now_time < 0):
                    now_time = 0

            cv2.imshow('motion', red)
            cv2.putText(frame_show, "MOTION", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_show, "TIMER: " + str(now_time), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('damn', frame_show)
        if cv2.waitKey(secs) == ord('q'):
            break
        frame_prev = to_gray(frame_now)