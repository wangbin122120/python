import numpy as np
import cv2

# 创建一个黑色背景图像：512*512像素，3个channelBGR。
img = np.zeros((512, 512, 3), np.uint8)

# 画一个5像素宽的对角直线，颜色为blue。
img = cv2.line(img, (0, 100), (200, 300), (255, 0, 0), 5)

cv2.imshow("draw line", img)
cv2.waitKey(0)
cv2.destroyAllWindows();