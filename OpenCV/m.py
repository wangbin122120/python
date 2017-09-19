
import cv2
import numpy as np

frame = cv2.imread('image/rice.png',cv2.IMREAD_GRAYSCALE)


laplacian = cv2.Laplacian(frame, cv2.CV_64F)
# cv2.cv_64f是数据类型。ksize是滤波器内核的大小。我们使用5，所以5x5区域作为滤波器。虽然我们可以使用这些梯度转换为纯边，我们也可以使用Canny边缘检测！
sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

# 提取边缘 ，Canny()
edges = cv2.Canny(frame,100,200)

cv2.imshow('Edges',edges)
cv2.imshow('Original', frame)
cv2.imshow('laplacian', laplacian)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()