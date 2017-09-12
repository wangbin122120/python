import numpy as np
from PIL import ImageGrab
import cv2
import time

from directkeys import ReleaseKey,PressKey, W, A, S, D

# 04-图片遮盖，遮盖区域用vertices一系列二维点表示
def roi(img,vertices):

    mask=np.zeros_like(img)
    # 凸填充，将mask中，把由vertices形成的点线区域填充为-255白色，原本是0漆黑一片。
    # 这样形成了mask一个多边形的挡板，0是挡板，要遮盖的区域，255是漏出来的区域。
    #  cv2.fillPoly 和 cv2.fillConvexPoly 在这里都一样， fillPoly()适用于非凸区域
    # cv2.fillConvexPoly(mask,vertices,255)
    cv2.fillPoly(mask, [vertices],255)

    # bitwise_and 对两个图片按pix做交集运算，次外还有_or ，_xor , _not
    # 把输入的图片img和挡板 mask 叠放，那么就遮盖了一部分的图片。
    masked = cv2.bitwise_and(img,mask)
    return masked


# 03-提取图像边缘
def process_img(original_image,threshold1=200,threshold2=300):
    processed_img=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    processed_img=cv2.Canny(processed_img,threshold1=200,threshold2=300)

    # 04-对提取边缘后的线条，遮盖掉一部分，
    # 使得汽车行驶的注意力集中在眼前部分，而不是远方或者天空，这些会影响行驶判断
    vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], np.int32)
    processed_img = roi(processed_img,vertices)
    return processed_img

# 02-实现一个界面监控功能
def screen_record():
    # gives us time to get situated in the game
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    last_time = time.time()
    while(True):

        # 给当前激活的窗口输入一个向前指令，控制方向
        PressKey(W)

        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(printscreen,threshold1=20,threshold2=20)
        # 现在考虑处理道路边缘，所以只输出边缘线即可。
        cv2.imshow('window',new_screen)
        # 用waitKey(20)延迟20ms刷新，达到每秒50帧
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()