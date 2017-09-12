import numpy as np
from grabscreen import grab_screen  # 09-获得屏幕窗口图片信息
import cv2
import time
from getkeys import key_check  # 09-获得键盘输入
import os

def keys_to_output(keys):
    # 预定义默认的结果是全部0
    output = [0, 0, 0]

    # 这里用 “IN” 是因为 如果同时按了 W + A/D 或者其他，那这时候的keys是一串字符。
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:  # 将w作为else ，也是一个技巧，大部分情况下都是w,所以哪怕判断错误，选择w大概率下较好。
        output[1] = 1
    return output

print('请开个无用的记事本用于键盘测试')
for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)


while True:
    keys = key_check()
    output = keys_to_output(keys)
    print(keys,output)
    time.sleep(0.5)