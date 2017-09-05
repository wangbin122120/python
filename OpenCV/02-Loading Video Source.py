import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 调用摄像头，如果有多个，就从0开始编号

while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
