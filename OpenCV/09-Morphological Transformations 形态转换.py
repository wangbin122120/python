# https://pythonprogramming.net/morphological-transformation-python-opencv-tutorial/?completed=/blurring-smoothing-python-opencv-tutorial/

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 如果帽子是蓝色的：
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])

    # 如果帽子是红色的
    # lower_red = np.array([30, 150, 50])
    # upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    
    # 从结果看 closing > dilation >> opening > erosion
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(mask, kernel, iterations=1)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Dilation', dilation)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()