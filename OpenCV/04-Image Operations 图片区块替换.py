# https://pythonprogramming.net/image-operations-python-opencv-tutorial/
import numpy as np
import cv2

img = cv2.imread('image/hat.png',cv2.IMREAD_COLOR)

# reference specific pixels

img[55,55]=[255,255,255] # 将某点变个颜色。

img[100:150,100:150] = [255,255,255] # 某块变个颜色

print(img.shape) # (161, 214, 3)
print(img.size)  # 103362
print(img.dtype) # uint8

img[0:74,0:87] = img[37:111,107:194] # 将某块的图片复制一份

cv2.imshow('?not chinese', img)
cv2.waitKey(0)
cv2.destroyAllWindows()