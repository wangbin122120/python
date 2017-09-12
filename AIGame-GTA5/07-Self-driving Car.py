import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from numpy import ones, vstack
from numpy.linalg import lstsq
from directkeys import ReleaseKey, PressKey, W, A, S, D
from statistics import mean


# 07-自动驾驶，定义几个操作键。
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)


def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


### 07- 原视频代码是不太一样了，因为他的lane是可以在同一侧的，所以这里不合适。因为我必须在左右两侧找参照点和线。
#  所以他就根据lane是否在同一侧，如果都是左侧，那就往左，否则右，一左一右就直走，貌似不太合理，但也能走走。
# if m1 < 0 and m2 < 0:
#     right()
# elif m1 > 0 and m2 > 0:
#     left()
# else:
#     straight()


# 07-我自己的想法，create by wangbin 2017-9-10
# 自动驾驶的简单算法，就是看人物/汽车所在位置离两边车道的距离，如果离左边太近（超过1.3倍）那么就走走右边，反之也是，否则直着走。
def self_driving(l_lane, r_lane):
    # 对于求点到线的距离，因为这里两个线段的头尾都是平行，所以y值不需要考虑，只需要求x的相对位置即可。
    car_x = 370  # 经过反复查看，人物中心点位置x=370
    l_lane_x = (l_lane[0] + l_lane[2]) / 2
    r_lane_x = (r_lane[0] + r_lane[2]) / 2

    print(l_lane_x,car_x,r_lane_x,l_lane,r_lane)

    if car_x <= l_lane_x:
        right()
    elif car_x >= r_lane_x:
        left()
    else:
        rate = (car_x - l_lane_x) / (r_lane_x - car_x)
        if rate > 1.3:
            left()
        elif rate < 1 / 1.3:
            right()
        else:
            straight()


# 06-画出车道，主要是找出左右两条分界线
def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker
        # (since we cannot assume the horizon will always be at the same point.)
        # 求得上下边界
        ys = []
        for i in lines:
            for ii in i:
                ys += [ii[1], ii[3]]  # 第 1,3分别是头尾两个点的纵向坐标y，所以这里ys是记录每线段的起止
        # 比如有10条线段，那么起止点有20个，相应的y值有20个，那就是这ys存储的信息

        min_y = min(ys)  # ys中最小的就是最上面的点，因为已经被凸遮盖了，所以不是0
        max_y = 600  # 定义600为最大，因为画框就这么大，所以无所谓。
        new_lines = []
        line_dict = {}

        # 求出每条edge线段的方程式，且延伸后与上下边界的交叉点。
        for idx, i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                # 对于每条线段，都解析出他们的方程式: y = m*x + b
                # 解析方法是numpy.linalg. lstsq，最小二乘法拟合出的一条直线。
                # 只有两个点的结果当然就是穿过这两点的直线拉。但是用拟合法求两点直线是否有点大材小用？
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                # 然后对每条线段都算出其与 上(min_y)下(max_y)两条水平横线交叉点的x值。
                x1 = (min_y - b) / m
                x2 = (max_y - b) / m

                # 然后保存好直线结构和上下两个交叉点坐标。
                line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        # 为了特征化和提炼数据，将所有相似的线段聚合一起，分出几个key线段，是个分类最优化的问题，这里用的方法是看m和b相差程度在20%以内，就属于同一个线段。
        # 这里面会有问题，1.就是20%这个阈值如何定？2.是线条分析的顺序会影响结果，因为先分析的线条总是做key,那后面分析的线条会被同化到先分析线条。
        # 是否可以考虑先随机的打乱线条的次序，或者用些梯度下降使得最终分类出的key最好。
        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:  # 初始化的时候，先丢一根 Lane 进去
                final_lanes[m] = [[m, b, line]]  # 这里用m作为key键，值为 方程式。

            else:
                found_copy = False

                for other_ms in final_lanes_copy:  # other_ms 取出的是 斜率m

                    if not found_copy:
                        # 将所有线段和 {final_lanes} 池子里的线段相似的合并在一起，
                        if abs(other_ms * 1.2) > abs(m) > abs(other_ms * 0.8):
                            if abs(final_lanes_copy[other_ms][0][1] * 1.2) > abs(b) > abs(
                                            final_lanes_copy[other_ms][0][1] * 0.8):
                                final_lanes[other_ms].append([m, b, line])
                                found_copy = True
                                break
                        # 不相似的单独开一条。键值是斜率m
                        else:
                            final_lanes[m] = [[m, b, line]]

        line_counter = {}

        # 对所有聚合分类好的key线段 ，依据相似线段的个数排序，取出最多的两个key，作为最终的车道，当然用的是其x,y的平均值难怪标出的车道都很烂，这里可以好好优化。
        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        ##### 自己修改的部分，create by wangbin 2017-9-10 00:01:56
        # 原始程序剩余部分：
        # top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
        # lane1_id = top_lanes[0][0]
        # lane2_id = top_lanes[1][0]
        # l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        # l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])
        #
        # return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]


        # 优化思路1：必须确保取出的车道是左边一条，右边一条，否则后面如何能控制？
        #           定义左右的方法，就是根据线段的中心点x是否在图片中轴线两侧。
        #           这里用斜率控制会更合理，并且还可以知道方向，过滤掉一些横向的杂乱无章的线条
        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1]
        for idx, l in enumerate(top_lanes):
            # x1, y1, x2, y2 = average_lane(final_lanes[l[0]])
            if l[0] > 1:  # 斜率大于1 基本可以认为是左边
                # if (x1 + x2) / 2 < 800 / 2:  # 因为视角和中心点的原因，效果不如斜率，当然两者和一起，条件太强，不容易出结果。
                # if l[0]>1 and (x1 + x2) / 2 < 800 / 2: # 这里800是画框的水平x方向pix数量，当然最好先全局预定义。
                left_lane = l[0]

                # print(idx,'左线段中间值为：',(x1 + x2) / 2 ,'个数',l[1])
                break

        for idx, l in enumerate(top_lanes):
            # x1, y1, x2, y2 = average_lane(final_lanes[l[0]])
            if l[0] < -1:
                # if (x1 + x2) / 2 > 800 / 2:
                # if l[0]<-1 and (x1 + x2) / 2 > 800 / 2:
                right_lane = l[0]
                # print(idx,'右线段中间值为：', (x1 + x2) / 2 ,'个数',l[1])
                break
        print('left:',left_lane,'right:',right_lane)
        return average_lane(final_lanes[left_lane]), average_lane(final_lanes[right_lane])

        # 当然这样优化的结果也不是最理想，可想而知，当拐弯的时候会比较麻烦。

    except Exception as e:
        print(str(e))
        return None,None


# 05-将lines画在img图片上，
def draw_lines(img, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            # 这里不返回图片，用cv2.line() 就已经作用到图片上，后续有显示图片即可。


# 04-图片遮盖，遮盖区域用vertices一系列二维点表示
def roi(img, vertices):
    mask = np.zeros_like(img)
    # 凸填充，将mask中，把由vertices形成的点线区域填充为-255白色，原本是0漆黑一片。
    cv2.fillPoly(mask, [vertices], 255)

    # bitwise_and 对两个图片按pix做交集运算，次外还有_or ，_xor , _not
    # 把输入的图片img和挡板 mask 叠放，那么就遮盖了一部分的图片。
    masked = cv2.bitwise_and(img, mask)
    return masked


# 06-处理图片中的车道。
def process_img_lane(original_image, edge_lines):
    try:
        l1, l2 = draw_lanes(original_image, edge_lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [255, 0, 0], 30) # 显示是红色
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
        return original_image, l1,l2
    except Exception as e:
        print(str(e))
        return original_image,None,None


# 03-处理图像边缘
def process_img_edge(original_image, threshold1=200, threshold2=300):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)

    # 05-高斯模糊模板,可以有效的出去图像的高斯噪声,这点和均值滤波，详细的还可以看：http://blog.csdn.net/on2way/article/details/46828567
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)

    # 04-对提取边缘后的线条，遮盖掉一部分，
    vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], np.int32)
    processed_img = roi(processed_img, vertices)

    # 05-通过霍夫变换寻找前面的边缘图 中的直线。
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, minLineLength=100, maxLineGap=20)

    if lines is not None:
        draw_lines(processed_img, lines)
        print('Lines found:', len(lines))
    else:
        print('No lines found!')

    return processed_img, lines


# 06-合并处理图像的边缘和图片中的车道。
def process_img(original_image, threshold1=200, threshold2=300):
    img_with_edge, edge_lines = process_img_edge(original_image, threshold1=threshold1, threshold2=threshold2)
    img_with_lanes, l_lane, r_lane = process_img_lane(original_image, edge_lines)
    return img_with_edge, img_with_lanes, l_lane, r_lane


# 02-实现一个界面监控功能
def screen_record():
    # gives us time to get situated in the game
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    last_time = time.time()
    while (True):

        # 给当前激活的窗口输入一个向前指令，控制方向
        # PressKey(W) # 05-draw line的时候先去除
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        # print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        # 06-程序升级，不仅画边缘线，还要画车道
        img_with_edge, img_with_lanes, l_lane, r_lane = process_img(printscreen, threshold1=200, threshold2=200)

        # 显示图片边缘线（反色）
        cv2.imshow('window', img_with_edge)
        # 显示原始图片带车道（左右2个）
        cv2.imshow('window2', cv2.cvtColor(img_with_lanes, cv2.COLOR_BGR2RGB))

        # 07- 通过判断car/man离两边车道的距离，尽量保持在中间行驶。
        if l_lane is not None and r_lane is not None:
            self_driving(l_lane, r_lane)
        else:
            straight()

        # 用waitKey(20)延迟20ms刷新，达到每秒50帧
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()
