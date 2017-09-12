import numpy as np
from PIL import ImageGrab
import cv2
import time

# 提取图像边缘
def process_img(original_image):
    # 这里将图片的颜色从3位的bgr转成1位的gray，这样更容易给后续的神经网络做处理。
    processed_img=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

    # cv2.canny()是用于寻找图像边缘的一种算法，具体可以看代码里面的描述和链接
    # http://en.wikipedia.org/wiki/Canny_edge_detector
    processed_img=cv2.Canny(processed_img,threshold1=200,threshold2=300)
    return processed_img

# 实现一个界面监控功能
def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(printscreen)
        cv2.imshow('window',new_screen)
        # cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        # 用waitKey(20)延迟20ms刷新，达到每秒50帧
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()