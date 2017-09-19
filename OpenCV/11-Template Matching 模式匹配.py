# https://pythonprogramming.net/template-matching-python-opencv-tutorial/?completed=/canny-edge-detection-gradients-python-opencv-tutorial/


import cv2
import numpy as np

# 在图片img_rgb 中查找   template ，模糊系数 threshold = 0.8 ,前提，template 的尺寸是没有变化，否则就很难查找。
img_rgb = cv2.imread('image/opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('image/opencv-template-for-matching.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)