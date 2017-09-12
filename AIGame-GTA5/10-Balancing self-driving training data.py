import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

# 10- balance_data.py


train_data = np.load('C:/Users/w/project/tmp/python/AIGame-GTA5/train/training_data.npy')

df = pd.DataFrame(train_data)  # @不知道是否数据包含多维，在给data 命名columns之后，内部程序get_loc(key)处理会报错。
print(df.head())
# DataFrame 是 pandas中的数据类型，像是二维的excel表格。详细：http://wiki.jikexueyuan.com/project/start-learning-python/311.html
# 因为之前没有预定义
#                                                    0          1
# 0  [[0, 228, 229, 229, 229, 230, 229, 229, 229, 2...  [0, 1, 0]
# 1  [[0, 230, 230, 232, 231, 232, 232, 232, 232, 2...  [0, 1, 0]
# 2  [[0, 231, 232, 232, 232, 233, 233, 233, 233, 2...  [0, 1, 0]
# 3  [[0, 231, 232, 233, 233, 233, 233, 233, 233, 2...  [0, 1, 0]
# 4  [[0, 232, 232, 234, 234, 234, 234, 234, 234, 2...  [0, 1, 0]


# 对数据进行统计， 关键字是 df[1] 也就是 key值。
print(Counter(df[1].apply(str)))  # Counter({'[0, 1, 0]': 7788, '[0, 0, 1]': 622, '[1, 0, 0]': 507})

lefts = []
rights = []
forwards = []

# 打乱 train_data 中数据的先后顺序，为了后续模型训练。
shuffle(train_data)

# 将数据分组
for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 1, 0]:
        forwards.append([img, choice])
    elif choice == [0, 0, 1]:
        rights.append([img, choice])
    else:
        print('no matches')

# 平衡各组数据量
########## 原程序如下：
# forwards = forwards[:len(lefts)][:len(rights)]
# lefts = lefts[:len(forwards)]
# rights = rights[:len(forwards)]
# 原程序打算对三组数据量做个平衡， 因为‘a','d'的数据量只有'w’的十分之一，这里其实可以用更容易理解的方式：
balace_len = min(len(lefts), len(rights), len(forwards))
forwards = forwards[:balace_len]
lefts = lefts[:balace_len]
rights = rights[:balace_len]

# 最终的测试数据由三组组成，最后再打乱顺序。
final_data = forwards + lefts + rights
shuffle(final_data)
# print(balace_len,balace_len*3,len(final_data)) # 500 1500 1500

# 数据另存，因为数据量减少的原因，所以整个文件的存储也相应减少
np.save('/Users/w/project/tmp/python/AIGame-GTA5/train/training_data_balanced.npy', final_data)
