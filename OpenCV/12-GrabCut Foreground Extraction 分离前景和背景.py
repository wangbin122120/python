# https://pythonprogramming.net/grabcut-foreground-extraction-python-opencv-tutorial/?completed=/template-matching-python-opencv-tutorial/

# 不是很智能的分离，只能分离出固定区域的图形相连图片。
# 背景替换的时候，简单相加会影响前景图片颜色。
import numpy as np
import cv2
from matplotlib import pyplot as plt

def pic_front():
    img = cv2.imread('image/opencv-python-foreground-extraction-tutorial.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (161,79,150,150)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img)
    plt.colorbar()
    plt.show()

def video_front(video_name='0',background_name=None):

    if video_name == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_name)

    if background_name is not None :
        # backgroud = cv2.imread(background_name,cv2.IMREAD_COLOR)
        backgroud = cv2.imread(background_name)[0:480,0:640,:]


    while True:
        _, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (161, 79, 300, 300)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        cv2.imshow('front', img)

        # print(img.shape,backgroud.shape)

        # 提取出前景图片后，将背景图片和前景图片融合，出来阿凡达一样的鬼。
        add_img = cv2.bitwise_and(img, backgroud, mask=mask2)
        cv2.imshow('merge front', add_img)

        # 提取出前景图片后，将背景图片替换成自己想要的背景。
        add_img = img + backgroud
        cv2.imshow('change backgroud', add_img)

        #TODO 前景和背景的完美融合,还没有实现
        pass

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


video_front('0','image/background_sky.jpg')