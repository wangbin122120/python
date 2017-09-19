# https://pythonprogramming.net/blurring-smoothing-python-opencv-tutorial/?completed=/color-filter-python-opencv-tutorial/

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

    cv2.imshow('Original',frame)
    cv2.imshow('hat', res)
    # 在分离出帽子后，但是图片中会有很多的颗粒点， 所以需要进行一点平滑处理，以去除噪点。
    # 在当前的设置中，考虑到这些参数较多，不容易调整。
    # 'bilateral Blur' 几乎没有平滑，没有去噪点，和原图相差无几。
    bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    cv2.imshow('bilateral Blur', bilateral)

    #其次 'Gaussian Blurring' 平滑效果适中，颗粒点少了，并且帽子上的文字图画还能保留清晰。
    blur = cv2.GaussianBlur(res, (15, 15), 0)
    cv2.imshow('Gaussian Blurring', blur)

    # 剩下两个平滑的就比较厉害了，median 基本是一坨了。
    median = cv2.medianBlur(res, 15)
    cv2.imshow('Median Blur', median)

    # 平均化 太模糊了。
    kernel = np.ones((15,15),np.float32)/225
    smoothed = cv2.filter2D(res,-1,kernel)
    cv2.imshow('Averaging',smoothed)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


