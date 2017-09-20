# https://pythonprogramming.net/loading-video-python-opencv-tutorial/?completed=/loading-images-python-opencv-tutorial/

# 补充，更多的视频操作参考:注意这底下几个链接程序是旧版的opencv2.4, 目前 3.0以上不支持。
# https://segmentfault.com/a/1190000003742481
# Python-OpenCV 处理视频（一）： 输入输出 https://segmentfault.com/a/1190000003804797
# Python-OpenCV 处理视频（二）： 视频处理 https://segmentfault.com/a/1190000003804807

import cv2
import numpy as np

print(cv2.__version__)  # 3.3.0

def CameraVideo(save=False):
    cap = cv2.VideoCapture(0)  # 调用摄像头，如果有多个，就从0开始编号

    if save == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        # 这里还可以将彩色变成灰色，同时输出做对比。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)

        # 可以将视频保存输出
        if save == True:
            out.write(gray)

        # 按下s键，保存当前摄像头图片
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # img_name=input('输入保存图片的名称，默认png格式：','image01.png')
            img_name = 'image01.png'
            cv2.imwrite(img_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    if save == True:
        out.release()

    cv2.destroyAllWindows()


def ShowVideo(videoName, gray=False):
    cap = cv2.VideoCapture(videoName)
    while True:
        ret, frame = cap.read()

        if gray == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(videoName, frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

CameraVideo()

# ShowVideo('output.avi')
# ShowVideo('video/studio-install-windows.mp4')

