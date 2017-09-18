import pandas as pd
import quandl


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

print('1.加载数据')
print(df.head())
print(df.tail())
print(df.shape)

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print('2.添加 pct 和 pct_change')
print(df.head())
print(df.tail())
print(df.shape)



forecast_col = 'Adj. Close'
df.fillna(-9999,inplace=True)   #如果存在空值，则用-9999填充。数据清洗。

# 将价格移动一下，用shift(-N) 将未来N个交易日的值拉回当前。 这里N=df/100=38，所以在最后的38个数字是空的。而前面不是空值，所以用这些作为预测值，以供学习。
import math
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)   # 我们将空值去除（也就是最后的空值啦，因为前面已经对空值做了-9999的操作）

print('3.新增一个预测价格的列label')
print(df.tail())
print(df.shape)
''' 所以这里展示的label是未来33个交易日的价格，
            Adj. Close    HL_PCT  PCT_change  Adj. Volume   label （未来33个交易日的价格）
Date                                                             
2017-07-25      969.03  0.794609   -0.172041    5793414.0  943.29  
2017-07-26      965.31  0.895049   -0.767902    2166225.0  946.65  
2017-07-27      952.51  1.785808   -1.720011    3685905.0  950.44   
2017-07-28      958.33  0.361045    1.090729    1795477.0  940.13 （2017-09-14）
2017-07-31      945.50  1.659704   -1.510417    2268160.0  935.29 （2017-09-15）
'''












