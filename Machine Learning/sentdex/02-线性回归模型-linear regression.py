# 1. 主要是说 预测模型要符合数据分布，预测效果才会准确，并且介绍了，线性回归模型的参数计算公式。
#  Regression - Theory and how it works
# https://pythonprogramming.net/simple-linear-regression-machine-learning-tutorial/?completed=/pickling-scaling-machine-learning-tutorial/
# https://www.youtube.com/watch?v=SvmueyhSkgQ&index=8&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v


# 2. 用python 计算线性回归模型的斜率 slope
# https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/?completed=/simple-linear-regression-machine-learning-tutorial/
from statistics import mean
import numpy as np

xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)


def best_fit_slope(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) ** 2) - mean(xs ** 2)))
    return m


m = best_fit_slope(xs, ys)
print(m)

# 3. 通过上面的介绍，继续计算 线性回归模型的 参数b,然后完成预测，并将训练数据和回归直线、预测结果都图像化。 best_fit_slope_and_intercept()
# How to program the Best Fit Line - Practical Machine Learning Tutorial with Python p.9
#
print('# 3. How to program the Best Fit Line')
from statistics import mean
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)

    return m, b


xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs]

predict_x = 7
predict_y = (m * predict_x) + b
print(m, b)
print(predict_x, predict_y)

style.use('ggplot')
# style.use('fivethirtyeight')  # 分别感受一下不同的图形界面。
plt.scatter(xs, ys, color='#003F72')
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()

# 4. 介绍R方 理论，用于评估回归模型的拟合程度。R Squared Theory - Practical Machine Learning Tutorial with Python p.10
# https://pythonprogramming.net/r-squared-coefficient-of-determination-machine-learning-tutorial/?completed=/how-to-program-best-fit-line-machine-learning-tutorial/
# https://youtu.be/-fgYp74SNtk?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
print('# 4. R Squared Theory - Practical Machine Learning Tutorial with Python p.10')

# 5. R方的python 计算实现： coefficient_of_determination()
# https://pythonprogramming.net/how-to-program-r-squared-machine-learning-tutorial/?completed=/r-squared-coefficient-of-determination-machine-learning-tutorial/
# https://youtu.be/QUyAFokOmow?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

from statistics import mean
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)

    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print('线性回归模型：\' y = %.4f * x + %.4f \' , R^2=%4f' % (m, b, r_squared))

predict_x = 7
predict_y = (m * predict_x) + b
print('预测 x = %f  -> y = %f ' % (predict_x, predict_y))

style.use('ggplot')
# style.use('fivethirtyeight')  # 分别感受一下不同的图形界面。
plt.scatter(xs, ys, color='#003F72')
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()

# 6. 随机生成数据集，进行测试。
# Testing Assumptions - Practical Machine Learning Tutorial with Python p.12
# https://youtu.be/Kpxwl2u-Wgk?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
# https://pythonprogramming.net/sample-data-testing-machine-learning-tutorial/?completed=/how-to-program-r-squared-machine-learning-tutorial/

print('# 6. Testing Assumptions - Practical Machine Learning Tutorial with Python p.12')
from statistics import mean
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

import random

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)

    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


# 6.
# hm 测试点数量，variance 随机波动范围，偏离直线最大距离，step 是相当于斜率，correlation 相关性，pos 斜率为正，neg向下，否则不相干，那么数据就完全是一个区域内随机生成。
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# 某种程度上说，
# xs, ys = create_dataset(40, 40, 2, correlation='pos')
xs, ys = create_dataset(40,10,2,correlation='pos')
# xs, ys = create_dataset(40,10,2,correlation='neg')
# xs, ys = create_dataset(40,10,2,correlation=False)
# 用自动生成是数据替代。
# xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print('线性回归模型：\' y = %.4f * x + %.4f \' , R^2=%4f' % (m, b, r_squared))

predict_x = 7
predict_y = (m * predict_x) + b
print('预测 x = %f  -> y = %f ' % (predict_x, predict_y))

style.use('ggplot')
# style.use('fivethirtyeight')  # 分别感受一下不同的图形界面。
plt.scatter(xs, ys, color='#003F72')
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()
