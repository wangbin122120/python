import numpy as np
from grabscreen import grab_screen  # 09-获得屏幕窗口图片信息
import cv2
import time
from getkeys import key_check  # 09-获得键盘输入
import os

# create_training_data.py
# 创建训练数据，不需要返回也不需要显示，只需要记录屏幕图片和操作信息这两个数据即可。
# 图片尺寸会被压缩成(160, 120)，当然数据是以点阵的形式保存。


# 将输入键 a,w,d 编码成 one hots 形式。
# A = [1,0,0]
# W = [0,1,0]
# D = [0,0,1]
def keys_to_output(keys):
    # 预定义默认的结果是全部0
    output = [0, 0, 0]

    # 这里用 “IN” 是因为 如果同时按了 W + A/D 或者其他，那这时候的keys是一串字符。
    # keys 是一个列表 ['A', 'D', 'W'] ,但哪怕同时按下多个，结果output也是一个 [1, 0, 0]
    # 因为在判断的优先级上是 A > D > else(W)
    # 所以 真正的键盘控制 vs 记录的键盘控制 & output：
    #         ['A', 'W']        A         [1, 0, 0]
    #         ['A', 'D', 'W']   A         [1, 0, 0]
    #         ['A', 'D']        A         [1, 0, 0]
    #         ['D', 'W']        D         [0, 0, 1]
    #         ['W']             W         [0, 1, 0]
    #                           附带上面的键盘控制的测试程序：“[09TEST]-keys_to_output.py”
    # @所以这个结果不是真实的结果，可以说是简化模型、突出重点的处理，所以后续优化可以对比下。
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:  # 将w作为else ，也是一个技巧，大部分情况下都是w,所以哪怕判断错误，选择w大概率下较好。
        output[1] = 1
    return output



file_name = 'C:\\Users\\w\\project\\tmp\\python\\AIGame-GTA5\\train\\training_data.npy'


# @程序这里的设定非常好：如果训练文件已经存在，那么会先加载这些数据，避免辛辛苦苦训练的数据被新训练的数据冲洗。
if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

# 训练数据：玩游戏啦！！！
def train_data():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while (True):

        if not paused:

            # 记录图像显示信息
            # 800x600 windowed mode
            screen = grab_screen(region=(0, 40, 800, 640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            # 尺寸得缩小一点，以便训练。 resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (160, 120))


            # 记录键盘输入信息
            keys = key_check()  # keys 是一个列表 ['A', 'D', 'W']
            output = keys_to_output(keys)  # 哪怕同时按下多个，结果也是一个 [1, 0, 0]

            # 没有跳过任何信息，把所有时间的图像和操作都记录下来
            training_data.append([screen, output])

            # 这里是每 1000帧保存一次，在训练过程中是不存在交互，实际记录的就是gta5的运行帧数。
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name, training_data)


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

                # 原本的程序中，暂停是没有保存的，这里添加一个保存
                print(len(training_data))
                np.save(file_name, training_data)

# 训练数据：玩游戏啦！！！如果只是显示 ，那可以不一定要训练
train_data()

# 保存了数据，自然要调用一下，看看结果如何：分辨率小了，颜色简单了，但信息都保留的很好，细节都在。
def show_train_data():
    # training_data = np.load(file_name)  # 程序开始设定就会加载，直接用就好。
    # for idx,data in enumerate(training_data[-1*show_last_frame:]): # @想指定显示最后一段，但np.load()是返回迭代器，所以不能直接指定最后一部分
    for idx, data in enumerate(training_data):
        img=data[0]
        press_key = data[1]
        cv2.imshow('show training data',img)
        print(idx,img.shape,press_key)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

show_train_data()