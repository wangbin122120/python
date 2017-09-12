import numpy as np
from PIL import ImageGrab
import cv2
import time

# 实现一个界面监控功能
def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        # 这里需要做颜色转换，否则三个颜色通道是错乱的
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        # 用waitKey(20)延迟20ms刷新，达到每秒50帧
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()