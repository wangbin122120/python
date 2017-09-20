#-*- coding: utf-8 -*-
# https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/?completed=/loading-video-python-opencv-tutorial/

import numpy as np
import cv2

def drawing_img(type):
    img = cv2.imread('image/hat.png',cv2.IMREAD_COLOR)
    if type == '直线':
        cv2.line(img,(0,0),(50,50),(255,255,255),15)

    elif type == '矩形':
        cv2.rectangle(img,(15,25),(200,150),(0,0,255),15)

    elif type == '圆形':
        cv2.circle(img,(100,63), 55, (0,255,0), -1)

    elif type == '多边形':
        pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        cv2.polylines(img, [pts], True, (0,255,255), 3)

    elif type == '文字':
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'OpenCV Tuts!',(0,130), font, 1, (200,255,155), 2, cv2.LINE_AA)

    cv2.imshow('?not chinese',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for type in ['直线','矩形','圆形','多边形','文字']:
    print('画个%s，按任意键继续。'%(type))
    drawing_img(type)
