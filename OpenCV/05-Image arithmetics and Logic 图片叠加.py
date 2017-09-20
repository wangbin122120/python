# https://pythonprogramming.net/image-arithmetics-logic-python-opencv-tutorial/?completed=/image-operations-python-opencv-tutorial/

import cv2
import numpy as np

# 500 x 250
imgs={}
imgs['image1'] = cv2.imread('image/3D-Matplotlib.png')
imgs['image2'] = cv2.imread('image/mainsvmimage.png')
imgs['add=100%image1 + 100%image2'] = imgs['image1']+imgs['image2']
imgs['weighted=60%image1 + 40%image2'] = cv2.addWeighted(imgs['image1'], 0.6, imgs['image2'], 0.4, 0)

for img in imgs:
    print('这个是图片：%s，尺寸%s，按任意键继续'%(img,imgs[img].shape))
    cv2.imshow(img,imgs[img])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


