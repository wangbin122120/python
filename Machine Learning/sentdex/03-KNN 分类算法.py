# 1. k-NN 分类算法 简介 ，knn是由待分类点所临近的K个点投票决定其所属分类。
# Classification w/ K Nearest Neighbors Intro - Practical Machine Learning Tutorial with Python
# https://youtu.be/44jq6ano5n0?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
# https://pythonprogramming.net/k-nearest-neighbors-intro-machine-learning-tutorial/?completed=/sample-data-testing-machine-learning-tutorial/

'''
# 2. k-NN 算法的python实现。
K Nearest Neighbors Application - Practical Machine Learning Tutorial with Python p.14
https://pythonprogramming.net/k-nearest-neighbors-application-machine-learning-tutorial/?completed=/k-nearest-neighbors-intro-machine-learning-tutorial/

数据下载链接：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/?C=S;O=A

这种乳腺癌数据库是从威斯康星大学医院取得的，数据下载完成后需要将列名添加到第一行
数据一共11列，包括：
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant) 这是分类结果
对应列名分别为：
id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses,class
数据样例：
1000025,5,1,1,1,2,1,3,1,1,2
1002945,5,4,4,5,7,10,3,2,1,2
1015425,3,1,1,1,2,2,3,1,1,2
1016277,6,8,8,1,3,4,3,7,1,2
1017023,4,1,1,3,2,1,3,1,1,2
'''
import pandas as pd
import numpy as np  # 用于将 get到的数据 序列化成 numpy.
from sklearn import preprocessing  # 数据标准化，归一化 ，参考：http://blog.csdn.net/dream_angel_z/article/details/49406573
from sklearn import neighbors  # 最临近算法
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分。

df = pd.read_csv('data/classification/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)  # 第一列id是无用干扰信息，必须抛去，否则结果非常差。

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)  # 0.978571428571 ，如果没有取出’id' ，结果只有0.5，完全就是抛硬币。所以在数据初始处理的时候，非常重要。

# 数据预测
example_measures = np.array([[5, 5, 1, 6, 3, 5, 3, 5, 5],
                             [2, 1, 1, 1, 1, 2, 3, 1, 1]])
print(clf.predict(example_measures))  # [4 2]

'''
# 3. 介绍 Euclidean 欧几里得距离是二阶范数，就是常见的两点距离，只不过是N维空间
# 矢量空间：https://wenku.baidu.com/view/c3d705fb770bf78a652954f6.html
# Euclidean Distance - Practical Machine Learning Tutorial with Python p.15
# https://youtu.be/hl3bQySs8sM?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
# https://pythonprogramming.net/euclidean-distance-machine-learning-tutorial/?completed=/k-nearest-neighbors-application-machine-learning-tutorial/
'''

from math import sqrt

plot1 = [1, 3]
plot2 = [2, 5]
euclidean_distance = sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)

'''
# 3. 

Creating Our K Nearest Neighbors Algorithm - Practical Machine Learning with Python p.16
Writing our own K Nearest Neighbors in Code - Practical Machine Learning Tutorial with Python p.17

https://pythonprogramming.net/programming-k-nearest-neighbors-machine-learning-tutorial/?completed=/euclidean-distance-machine-learning-tutorial/
https://youtu.be/n3RqsMz3-0A?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],
           'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color=i)
# 等价于：
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)
#
plt.show()


'''
# 4. 自定义 knn算法，实现 平面二维点的分类。

https://pythonprogramming.net/coding-k-nearest-neighbors-machine-learning-tutorial/?completed=/programming-k-nearest-neighbors-machine-learning-tutorial/
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter  # 统计个数
from numpy import linalg as LA  # 求

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]],
           'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [2, 7]


# 自定义一个knn函数，演示内部原理
def k_nearest_neightbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K 值太小，应该比分类种类大')
    distances = []
    # 记录与各点距离及其分类
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))  # 这个求两个数据点之间的欧氏距离。比自己写函数速度快很多。
            # norm(x, ord=None, axis=None, keepdims=False) 求范数，默认阶数 ord = 2 ，就是欧氏距离 ；ord=np.inf 无穷范数	max(|xi|) 	；ord=1	一范数|x1|+|x2|+…+|xn|
            distances.append([euclidean_distance, group])  # 记录与各点距离及其分类

    print(distances)
    '''
            [[5.0990195135927845, 'k'],
             [4.0, 'k'],
             [6.0827625302982193, 'k'], 
             [4.4721359549995796, 'r'], 
             [5.0, 'r'], 
             [6.0827625302982193, 'r']]
    '''
    votes = [i[1] for i in sorted(distances)[:k]]  # 初步筛选：排序后选取最大的前k个
    vote_result = Counter(votes).most_common(1)[0][0]  # 对 筛选后统计票数，取最多票
    print(Counter(votes))                   # Counter({'r': 2, 'k': 1})
    print(Counter(votes).most_common(1))    # [('r', 2)] ，1表示统计最多的前1个。
    print(Counter(votes).most_common(1)[0])  # ('r', 2)
    print(vote_result)                       # r



k_nearest_neightbors(dataset, new_features, k=3)
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color=i)
## 等价于：
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)
#
plt.show()


'''
# 5. 用前面的乳腺癌数据进行测试自己写的knn算法，区别于 #4. 的地方在于数据装载，将numpy转换为字典类型。  #4. + 数据 = #5. = #2. 
前面用sklearn 中的 knn算法，运行速度很快，几乎是自己编写程序的10倍。
https://pythonprogramming.net/testing-our-k-nearest-neighbors-machine-learning-tutorial/?completed=/coding-k-nearest-neighbors-machine-learning-tutorial/
https://pythonprogramming.net/final-thoughts-knn-machine-learning-tutorial/?completed=/testing-our-k-nearest-neighbors-machine-learning-tutorial/
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('data/classification/breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])   # 最后一位是class，注意别添加进数据集

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# 4. 中的数据处理方式在分类模型中很实用。
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total) # Accuracy: 1.0 准确度高的可怕

