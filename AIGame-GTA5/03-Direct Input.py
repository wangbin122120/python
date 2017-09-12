import numpy as np
from PIL import ImageGrab
import cv2
import time


# 这里提到了pyautogui，但不使用。应该是有些原因。
# import pyautogui
# print('down')
# pyautogui.keyDown('w')
# time.sleep(3)
# print('up')
# pyautogui.keyUp('w')

# directkeys.py中保存了如何通过电脑生成键盘指令。
# 我们在程序中输入控制，然后程序返回给游戏键盘指令。
from directkeys import ReleaseKey,PressKey, W, A, S, D


# 提取图像边缘
def process_img(original_image):
    # 这里将图片的颜色从3位的bgr转成1位的gray，这样更容易给后续的神经网络做处理。
    processed_img=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

    # cv2.canny()是用于寻找图像边缘的一种算法，具体可以看代码里面的描述和链接
    # http://en.wikipedia.org/wiki/Canny_edge_detector
    # https://www.kancloud.cn/aollo/aolloopencv/271603
    processed_img=cv2.Canny(processed_img,threshold1=200,threshold2=300)
    return processed_img

# 实现一个界面监控功能
def screen_record():

    # gives us time to get situated in the game
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    last_time = time.time()
    while(True):

        # 给当前激活的窗口输入一个向前指令，控制方向
        PressKey(W)
        PressKey(W)
        PressKey(W)
        time.sleep(0.5)
        PressKey(A)
        PressKey(W)
        time.sleep(0.5)
        PressKey(D)
        PressKey(W)
        # 但这样用sleep()的坏处是同步的显示输出会停顿
        # 需要用到其他函数处理
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(printscreen)
        #cv2.imshow('window',new_screen)
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        # 用waitKey(20)延迟20ms刷新，达到每秒50帧
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()