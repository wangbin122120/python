# https://pythonprogramming.net/canny-edge-detection-gradients-python-opencv-tutorial/?completed=/morphological-transformation-python-opencv-tutorial/

'''
介绍图像梯度和边缘检测。图像梯度可以用来测量方向的强度，边缘检测完全像它听起来的那样：它发现边缘！
关于 canny的原理，可以参考如下：
http://blog.csdn.net/liuzhuomei0911/article/details/51345591


Canny边缘检测：
原理：1986年由John F.Canny提出的算法
步骤:
1噪声去除
由于边缘检测容易受到噪声影响，因此需去除噪声，第一步是使用5*5的高斯滤波器
（先看5*5高斯滤波器）

2计算图像梯度
对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数（图像梯度）
(Gx和Gy)。根据得到的这两幅梯度图(Gx和Gy)找到边界的梯度和方向，公式如下:
Edge_Gradient(G)=sqrt( Gx*Gx + Gy*Gy )
Angle( C塔 ) = tan-1(上标)(Gx/Gy)
梯度的方向一般总是与边界垂直。梯度方向有四类：垂直，水平，和两个对角线
（算子和梯度也需要看）

3非极大值抑制
获得梯度的方向和大小后，对整幅图像做扫描，去除非边界上的点。对每个像素检查，
看这个点的梯度是不是周围具有相同梯度方向中的点中最大的

4滞后阈值
需要设置两个阈值：minVal和maxVal。当图像的灰度梯度高于maxVal时被认为是真的
边界，低于minVal的边界会被抛弃。介于两者之间的话，需要看该点是否与某个被确定
为真正的边界点相连，如果是就认为它是边界点，否则就抛弃。
边界是长的线段。
'''

import cv2
import numpy as np

def video_edge(video_name):

    if video_name == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_name)

    while True:

        # Take each frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 如果帽子是蓝色的：
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])

        # 如果帽子是红色的
        # lower_red = np.array([30, 150, 50])
        # upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        # cv2.cv_64f是数据类型。ksize是滤波器内核的大小。我们使用5，所以5x5区域作为滤波器。虽然我们可以使用这些梯度转换为纯边，我们也可以使用Canny边缘检测！
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

        # 提取边缘 ，Canny()
        edges = cv2.Canny(frame,100,200)

        cv2.imshow('Edges',edges)
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('laplacian', laplacian)
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('sobely', sobely)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def pic_edge(pic_name):
    frame = cv2.imread(pic_name,cv2.IMREAD_GRAYSCALE)
    # 提取边缘 ，Canny()
    edges = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edges)
    cv2.imshow('Original', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # video_edge(0)
    # video_edge('video/studio-install-windows.mp4')
    pic_edge('image/rice.png')