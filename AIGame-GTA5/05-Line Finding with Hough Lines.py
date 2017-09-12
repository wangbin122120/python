import numpy as np
from PIL import ImageGrab
import cv2
import time

from directkeys import ReleaseKey,PressKey, W, A, S, D

### 这次程序会在之前的基础上去寻找 行驶过程中道路两边的线条，作为判断行驶方向的依据。


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

# 05-将lines画在img图片上，
# lines:每条直线的起始和结束坐标：( (x1, y1, x2, y2),(x1, y1, x2, y2),...,(x1, y1, x2, y2))
def draw_lines(img,lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    # 这里不返回图片，用cv2.line() 就已经作用到图片上，后续有显示图片即可。
    # 注意，（x,y） ，x 是横向的列， y 是 纵向的行。
    # 关于如何用opencv绘制几何图形的例子，查看：http://blog.topspeedsnail.com/archives/2071


# 03-提取图像边缘
def process_img(original_image,threshold1=200,threshold2=300):
    processed_img=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    processed_img=cv2.Canny(processed_img,threshold1=200,threshold2=200)

    # 05-高斯模糊模板,可以有效的出去图像的高斯噪声,这点和均值滤波，详细的还可以看：http://blog.csdn.net/on2way/article/details/46828567
    processed_img=cv2.GaussianBlur(processed_img,(5,5),0)

    # 04-对提取边缘后的线条，遮盖掉一部分，
    # 使得汽车行驶的注意力集中在眼前部分，而不是远方或者天空，这些会影响行驶判断
    # vertices_far = np.array([[10, 600], [10, 200], [200, 100], [600, 100], [800, 200], [800, 600]], np.int32)
    vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], np.int32)
    processed_img = roi(processed_img,vertices)

    # 05-通过霍夫变换寻找前面的边缘图 中的直线。
    # HoughLines()是霍夫变换，为了检测图像中的直线，可以根据点和线查找出其中的直线。
    # HoughLinesP()是概率霍夫变换，检测图片中分段的直线，
    # cv2.HoughLinesP(edges, 变换的查找过程是像一个移动的旋转棍，
    #       1,              半径步长为1， 每次旋转棍中心点的移动步长，
    #       np.pi / 180,    角度步长为 pi/180， 每次旋转的角度
    #       100,            通过某个图像点的最大次数（阈值）
    #       minLineLength, 检测最短直线长度，太短的就不算
    #       maxLineGap)    最大直线间隙、缺口，间隙小的线段算同一条，否则分开算
    # 返回值：每条直线的起始和结束坐标：( (x1, y1, x2, y2),(x1, y1, x2, y2),...,(x1, y1, x2, y2))
    # 更多霍夫变换的例子看程序[05]-HoughLinesP_street.py
    lines=cv2.HoughLinesP(processed_img,1,np.pi/180,100,minLineLength=50,maxLineGap=10)

    if lines is not None:
        draw_lines(processed_img,lines)
        print('Lines found:',len(lines))
    else:
        print('No lines found!')

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
        # PressKey(W) # 05-draw line的时候先去除

        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        # print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(printscreen,threshold1=200,threshold2=200)
        # 现在考虑处理道路边缘，所以只输出边缘线即可。
        cv2.imshow('window',new_screen)
        # 用waitKey(20)延迟20ms刷新，达到每秒50帧
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()