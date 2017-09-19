import pandas as pd
import quandl  # 用于下载金融数据
import numpy as np  # 用于将 get到的数据 序列化成 numpy.
from sklearn import preprocessing  # 数据标准化，归一化 ，参考：http://blog.csdn.net/dream_angel_z/article/details/49406573
from sklearn import svm  # 支持向量机，svm训练和预测模型 http://blog.csdn.net/v_july_v/article/details/7624837
from sklearn.linear_model import LinearRegression  # 线性回归，在本次训练中效果较好。
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分。
import datetime  # 用于绘制plot()图的横坐标
import matplotlib.pyplot as plt  # 绘图
from matplotlib import style  # 绘图
import pickle  # 绘图

'''
本程序仅仅是提供了sklearn 如何训练和预测数据，并且对结果进行展示，这个例子并不能实际用于生产。
因为金融股票数据的波动，是受到各种因素的影响，并且每类每个公司的影响因素和受影响程度是非常不同。

教程文字版：https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/

'''

# 1. 下载数据，并且定义了两个辅助列，定义用于数据集合。
df = quandl.get(dataset='WIKI/GOOGL', api_key='6yWSw4q4iEWbsU-SspQ5')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# 2. 制造预测结果。
forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)  # 如果存在空值，则用-9999填充。数据清洗。
forecast_out = int(0.1 * len(df))
print('一共%d个交易日，预测%d个' % (len(df), forecast_out))
last_ = df[forecast_col][-forecast_out:]
df['label'] = df[forecast_col].shift(-forecast_out)


# 3. 数据标准化，划分训练和测试数据。
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)  # 标准化（Standardization）， (x-均值)/方差

X_lately = X[-forecast_out:]

X = X[:-forecast_out]
df.dropna(inplace=True)  # 我们将空值去除（也就是最后的空值啦，因为前面已经对空值做了-9999的操作）
y = np.array(df['label'])
print(len(X),len(y))
df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. training and evaluation
# clf = svm.SVR()           # 用svm模型 confidence=0.84,实际画图差异挺大。
clf = svm.SVR(kernel='linear')  # svm 中的线性模型结果也很不错，confidence : 0.976
# clf = LinearRegression()    # 用线性模型训练 confidence=0.97
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)  # 1 -  ((y_true - y_pred) ** 2).sum() /  ((y_true - y_true.mean()) ** 2).sum()
# 或者从保存的训练模型直接恢复，那么不需要定义、训练，直接就可以用于预测：
# clf = pickle.load(open('01-金融数据训练和预测.pickle', 'rb'))

forecast_set = clf.predict(X_lately)
print('confidence :', confidence)  # 这里如果叫准确率有点不妥，毕竟不是分类模型。
# for i in range(len(forecast_set)):
#     print(X_lately[i], forecast_set[i], last_[i])

# 5. 绘图
style.use('ggplot')  # 美化输出结果，而且还很方便的使用放大工具
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()  # 这段是之前的走势。
df['Forecast'].plot()  # 这段是预测走势
last_.plot()  # 这段是真实的走势
print(df)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 6. 保存训练结果，下次直接调用就可以做预测，一般在预测结束之后，预测开始之前进行保存。
# 保存
with open('01-金融数据训练和预测.pickle', 'wb') as f:
    pickle.dump(clf, f)
# # 提取
# pickle_in = open('01-金融数据训练和预测.pickle', 'rb')
# clf_2 = pickle.load(pickle_in)
# # 预测，并且结果和之前的一至
# forecast_set2 = clf_2.predict(X_lately)
# print(forecast_set == forecast_set2)
