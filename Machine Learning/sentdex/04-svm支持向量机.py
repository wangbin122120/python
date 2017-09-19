import numpy as np
'''
svm ( Support Vector Machine) The objective of the Support Vector Machine is to find the best splitting boundary between data
支持向量机的目标是寻找数据之间的最佳分裂边界。 在60-90年代都是最好的机器学习算法。
支持向量机的原理：
    1. 先确定支持向量
        
https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/?completed=/final-thoughts-knn-machine-learning-tutorial/
'''

import pandas as pd
import numpy as np  # 用于将 get到的数据 序列化成 numpy.
from sklearn import preprocessing  # 数据标准化，归一化 ，参考：http://blog.csdn.net/dream_angel_z/article/details/49406573
from sklearn import neighbors  # 最临近算法
from sklearn import svm       # svm 算法
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分。


df = pd.read_csv('data/classification/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)  # 第一列id是无用干扰信息，必须抛去，否则结果非常差。

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC()  # 用sklearn计算很方便，修改一下接口方法即可。

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)  # svm:0.95 , knn :0.978571428571 ，如果没有取出’id' ，结果只有0.5，完全就是抛硬币。所以在数据初始处理的时候，非常重要。
# svm 算法速度比knn快几倍，效果也很好。

# 数据预测
example_measures = np.array([[5, 5, 1, 6, 3, 5, 3, 5, 5],
                             [2, 1, 1, 1, 1, 2, 3, 1, 1]])
print(clf.predict(example_measures))  # [4 2]
