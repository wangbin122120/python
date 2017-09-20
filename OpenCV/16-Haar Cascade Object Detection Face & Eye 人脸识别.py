# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/?completed=/mog-background-reduction-python-opencv-tutorial/

'''
引入haarcascades辅助识别人脸和眼睛。虽然使用很简单， 但是背后的算法和原理非常复杂，建议参考一些资料。
另外在opencv的下载安装包里面，有附带其他几个xml。

更多opencv 中的haar adaboost 介绍，可以看博客：

2015年（算是旧的了）：http://blog.csdn.net/zy1034092330
    需要一定的基础，有空认真看看其中的：
    OpenCV中的Haar+Adaboost（一）：Haar特征详细介绍
        http://blog.csdn.net/zy1034092330/article/details/48850437
    OpenCV中的Haar+Adaboost（七）：分类器训练过程
        http://blog.csdn.net/zy1034092330/article/details/54600014

'''


import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('image/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('image/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()