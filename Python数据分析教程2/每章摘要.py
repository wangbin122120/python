

-----------------
第三章：

#初始化
np.eye(n)      		#单位1矩阵 nxn
np.arange(n)   		#数组range 
np.zeros(N)		#零数组	   
np.ones(N)		#1数组

#文件读写操作
loadtxt(文件名)		# 提取数据保存到数组
numpy.loadtxt(文件名, dtype=数据类型, comments=’#’, delimiter=分隔符, converters={列：转换函数}, skiprows=0, usecols=(列标), unpack=分段输出, ndmin=0)
savetxt(文件名,数组)  	# 保存数组数据到文件



#计算
np.average(c,weights=v) #加权平均。金融指标:成交量加权平均价格（VWAP） = np.sum(c*v) / np.sum(v)
np.mean(c)		#算术平均
np.sum(c)		#求和
np.min(l)		#数组极值
np.max(h)		#数组极值
np.argmax(averages)	#数组极值所在位置
np.argmin(averages)	#数组极值所在位置
np.ptp(h)		#数组幅度=最大-最小值
np.median(c)		#中位数，若数组个数是偶数，那么中位数是中间两个值的平均
np.msort(c)		#数组排序
np.var(c)		#方差
np.std(c)		#标准差。金融指标：风险度量 np.std(returns)
np.diff(c)		#间隔差，数组元素(n) - 数组元素(n-1)。
			#金融指标：简单收益率 returns = np.diff(c) / c[:-1]
			#金融指标：对数收益率 logreturns = np.diff(np.log(c))
				# log收益率 = log(前一日收盘价/今日收盘价)  = log(前一日收盘价) -log(今日收盘价)
np.sqrt()		#平方根。金融指标：年波动率 np.std(logreturns)/np.mean(logreturns)/np.sqrt(1/252)
np.where(returns>0)	#数组筛选
np.ravel(np.where(dates==0)) 	#where 之后的数组常常用ravel做扁平化
np.split(weeks_indices,3) 	#数组分割
np.apply_along_axis(summarize, 1, weeks_indices,open, high, low, close)	
		#numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)
		#对于数据arr, *args, **kwargs中的每一行（axis=1）或每一列（axis=0）作用于函数func1d（），并分别返回结果

np.convolve(a,b,'full')    	# 卷积：default:乘积的所有结果，个数：(N+M-1,)
np.convolve(a,b,'same')    	# 卷积：任意两层有相交的结果，个数：max(M, N)
np.convolve(a,b,'valid')   	# 卷积：所有层都相交的结果，两者做卷积完全重叠的区域，个数：max(M, N) - min(M, N) + 1
np.linspace(a , b , N)	#分割，对[a,b]区间分成N份，np.linspace(0,1,3)=array([ 0. ,  0.5,  1. ])
np.dot(b, x)		#矩阵乘积。
a.clip(min,max)		#对数组进行修剪，小于min的值设置为min值，对大于max值同样处理。
a.compress(a > 2)	#对数组进行筛选，保留满足条件的值。
b.prod()		#计算所有元素的乘积，有一个为0，则结果一定为0
b.cumprod()		#迭代每个元素乘积的结果，np.array([1,2,3,4,5]).cumprod() = array([  1,   2,   6,  24, 120])


#关于时间日期的转换
import datetime
return datetime.datetime.strptime(s, "%d-%m-%Y").date().weekday() #将字符串转换成日期，weekday()返回的是星期


#关于plot的设置
import matplotlib.pylab as plt
%matplotlib inline
t = np.arange(N - 1, len(c))
plt.figure(figsize=(8,4))
plt.plot(t,c[N-1:],"b--",lw=1.0,label="c")
plt.plot(t,ema,"b--",color='red',lw=2.0,label="ema_"+str(N))
plt.plot(t,sma,"b--",color='yellow',lw=2.0,label="sma_"+str(N))
plt.legend()
plt.show()


