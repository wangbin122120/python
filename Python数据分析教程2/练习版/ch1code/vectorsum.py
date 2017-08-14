# !/usr/bin/env python
# -*- coding: utf-8 -*
# ----------------------------------------------------------------------- #
# 任务名：
# 需求人员:  wangbin
# 联系方式:  wangbin122120@163.com
# 运行环境： Windows 10   + Anaconda3_4.4.0(python3.6)
# ----------------------------------------------------------------------- #
# 版本　 脚本修改人     修改日期            修改内容
# V1.0   wangbin      2017-07-17 00:16:06    create
#
#
#
#
# ----------------------------------------------------------------------- #

'''
1. 程序目的、需求、疑问：

    a. # 1.测试函数运行结果

    b. 对比python和numpy运行效率差异

    c.

2. 解决方案、伪代码、结论：

    a. 本章小结:
        在本章中，我们安装了NumPy以及其他推荐软件。我们成功运行了向量加法程序，并以此证
        明了NumPy优异的性能。随后，我们介绍了交互式shell工具IPython。此外，我们还列出了供你参
        考的NumPy文档和在线资源。

    b. run result: numpy 比 python 内置运行速度快了2倍啊!
因为:NumPy的大部分代码都是用C语言写成的，这使得NumPy比纯Python代码高效得多。


    c.
        # 不难想象,之前的差异100的原因在于,python3中int的类型永不溢出,为了防止溢出做了很多额外的工作.
        # 但对于原来错误的程序来说,由于没有设置类型,则少了这部分的时间,所以是100倍(应该是直接堆C语言上去了).所以2倍是相对客观的!
        # 所以arange()产生数据时,注意使用的过程中是否会超长,如果为了保证正确性,可以先设置类型dtype=object,在确定正确结果并在优化时对于数据类型进行放开,这样可以加速不少.



3. 程序优化建议：

    a. 在写代码进行运行效率比较的时候,首先要对结果的正确性进行验证

    b. # 结论: numpy中的 int类型的计算溢出:因为是C语言写的原因?
        # 但实际上问题的根源是在arange()的使用上面:
        # 修改原代码中的arange()如果没有设置dtype= object就会溢出!
        def sum_numpy(n):
            a = np.arange(n, dtype=object) ** 2
            b = np.arange(n, dtype=object) ** 3
            c = a + b
            return c



    c. 如果没有上述的深入研究和探索,应该不知道,上述那么多结论!很棒!


4. 程序使用说明、运行结果：

    a. 使用说明：直接执行

    b. 运行时间： < 1 秒

    c. 执行结果：


'''

# --------- 0. 常用类库引用区---------
# import math
# import matplotlib.pyplot as plt
import numpy as np
# import os
# import pandas as pd  # 能够快速便捷处理结构化数据的数据结构和函数
# from scipy.misc import imread, imsave, imresize
# from scipy.spatial.distance import pdist, squareform
import sys
from time import time


# --------- 1. 函数及类预定义区---------
def sum_python(n):
    a = [x ** 2 for x in range(n)]
    b = [x ** 3 for x in range(n)]
    return [a[i] + b[i] for i in range(n)]


'''
#没有指定arange的数据类型,则会发生溢出问题.
def sum_numpy(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    return a + b
'''


def sum_numpy(n):
    a = np.arange(n, dtype=object) ** 2
    b = np.arange(n, dtype=object) ** 3
    return a + b


# --------- 2. 数据及逻辑加工区---------

'''
假设我们需要对两个向量a和b做加法。这里的向量即数学意义上的一维数组，随后我们将
在第5章中学习如何用NumPy数组表示矩阵。向量a的取值为0~n的整数的平方，例如n取3时，向
量a为0、 1或4。向量b的取值为0~n的整数的立方，例如n取3时，向量b为0、 1或8。用纯Python
代码应该怎么写呢？我们先想一想这个问题，随后再与等价的NumPy代码进行比较
'''

# 1.测试函数运行结果

# print(sum_python(3))
#
# print(type(sum_python(3)))


# print(sum_numpy(3))
#
# print(type(sum_numpy(3)))


'''
下面我们做一个有趣的实验。在前言部分我们曾提到， NumPy在数组操作上的效率优于纯
Python代码。那么究竟快多少呢？接下来的程序将告诉我们答案，它以微秒（106 s）的精度分别
记录下numpysum()和pythonsum()函数的耗时。这个程序还将输出加和后的向量最末的两个元
素。让我们来看看纯Python代码和NumPy代码是否得到相同的结果：
'''
# 2. 对比python和numpy运行效率差异

n = int(sys.argv[1])
stime = time()
sp = sum_python(n)
print(time() - stime)

stime = time()
sn = sum_numpy(n)
print(time() - stime)

'''
run result: numpy 比 python 内置运行速度快了100倍啊!(后面发现,这个结果是错误的!)
因为:NumPy的大部分代码都是用C语言写成的，这使得NumPy比纯Python代码高效得多。 

D:\>D:\Anaconda3\python.exe D:/wangbin/project/python/Python数据分析教程2/练习版/ch1code/vectorsum.py 1000000
1.1477932929992676
0.011507749557495117
'''

# 3.验证结果的正确性
print(np.array(sp)[:10], sn[:10])
print(np.array(sp)[-10:-1], sn[-10:-1])
print('两个结果的差异:', np.sum(np.array(sp) - sn))

'''
但在结果正确性验证的时候,发现sn的数据并不完全相同
D:\>D:\Anaconda3\python.exe D:/wangbin/project/python/Python数据分析教程2/练习版/ch1code/vectorsum.py 1000
0.003002166748046875
0.0
两个结果的差异: 0

D:\>D:\Anaconda3\python.exe D:/wangbin/project/python/Python数据分析教程2/练习版/ch1code/vectorsum.py 10000
0.01050567626953125
0.0
两个结果的差异: 2499430448103424
'''

# 通过几次缩小输入值n ,发现从1291开始不一致,将不一致的位置和值分别输出:
print([(i, sp[i], sn[i]) for i in range(n) if sp[i] != sn[i]])

'''
D:\>D:\Anaconda3\python.exe D:/wangbin/project/python/Python数据分析教程2/练习版/ch1code/vectorsum.py 1292
0.0010018348693847656
0.0
[  0   2  12  36  80 150 252 392 576 810] [  0   2  12  36  80 150 252 392 576 810]
[2108641292 2113578276 2118522960 2123475350 2128435452 2133403272
 2138378816 2143362090 2148353100] [ 2108641292  2113578276  2118522960  2123475350  2128435452  2133403272
  2138378816  2143362090 -2146614196]
两个结果的差异: 8589934592
[(1290, 2148353100, -2146614196), (1291, 2153351852, -2141615444)]
'''


# 结论: numpy中的 int类型的计算溢出:因为是C语言写的原因?
# 但实际上问题的根源是在arange()的使用上面:
# 修改原代码中的arange()如果没有设置dtype= object就会溢出!
def numpysum(n):
    a = np.arange(n, dtype=object) ** 2
    b = np.arange(n, dtype=object) ** 3
    c = a + b
    return c


# 修改后运行速度就没有那么大的差异了.仅2倍.
'''
D:\>D:\Anaconda3\python.exe D:/wangbin/project/python/Python数据分析教程2/练习版/ch1code/vectorsum.py 10000000
11.897812843322754
6.4665563106536865
'''

# 不难想象,之前的差异100的原因在于,python3中int的类型永不溢出,为了防止溢出做了很多额外的工作.
# 但对于原来错误的程序来说,由于没有设置类型,则少了这部分的时间,所以是100倍(应该是直接堆C语言上去了).所以2倍是相对客观的!
# 所以arange()产生数据时,注意使用的过程中是否会超长,如果为了保证正确性,可以先设置类型,在确定正确结果并在优化时对于数据类型进行放开,这样可以加速不少.


