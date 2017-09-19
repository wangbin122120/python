# https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/?completed=/image-arithmetics-logic-python-opencv-tutorial/



# 这里介绍阈值，
import cv2
import numpy as np

img = cv2.imread('image/bookpage.jpg')
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('original',img)


# 经过反色后的效果看起来还不如用threshold仔细调整的好。
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('th250', th)

for i in range(20):
    p = i*3
    print(p)
    # threshold 有两个阈值：minVal和maxVal。当图像的灰度梯度高于maxVal时被认为是真的边界，低于minVal的边界会被抛弃。介于两者之间的话，需要看该点是否与某个被确定为真正的边界点相连，如果是就认为它是边界点，否则就抛弃。边界是长的线段。
    retval, threshold255 = cv2.threshold(img, p, 255, cv2.THRESH_BINARY)  # p=6的时候比较好。
    retval, threshold120 = cv2.threshold(img, p, 120, cv2.THRESH_BINARY)
    retval, threshold_gray = cv2.threshold(grayscaled, p, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('threshold255',threshold255)
    cv2.imshow('threshold120', threshold120)
    cv2.imshow('threshold_gray', threshold_gray)

    k = cv2.waitKey(0) & 0xFF  # esc键
    # ASCII中27是esc
    if k == 27:
        break


cv2.destroyAllWindows()