# https://lizonghang.github.io/2016/07/25/%E9%9C%8D%E5%A4%AB%E5%8F%98%E6%8D%A2%E6%A3%80%E6%B5%8B%E7%9B%B4%E7%BA%BF/
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# 霍夫变换
def HoughLines():
    import cv2
    import numpy as np
    im = cv2.imread('image/[05]-HoughLinesP_street.jpg')
    im = cv2.GaussianBlur(im, (3, 3), 0)
    edges = cv2.Canny(im, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 260)
    result = im.copy()
    for line in lines[0]:
        rho = line[0]
        theta = line[1]
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
            pt1 = (int(rho / np.cos(theta)), 0)
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            cv2.line(result, pt1, pt2, (0, 0, 255))
        else:
            pt1 = (0, int(rho / np.sin(theta)))
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            cv2.line(result, pt1, pt2, (0, 0, 255), 1)
    cv2.imshow('Hough', result)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


# 概率霍夫变换
def HoughLinesP():
    import cv2
    import numpy as np
    im = cv2.imread('image/[05]-HoughLinesP_street.jpg')
    im = cv2.GaussianBlur(im, (3, 3), 0)
    edges = cv2.Canny(im, 50, 150, apertureSize=3)
    result = im.copy()
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength, maxLineGap)
    # 我们可以把这些线段都画出来更好看
    for l in lines:
        print(len(l))
        for x1, y1, x2, y2 in l:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('HoughP', result)
        cv2.waitKey(500)

    cv2.destroyAllWindows()


HoughLines()
