import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  ### 注意用tf.Variable
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases  # 预测y的值

loss = tf.reduce_mean(tf.square(y - y_data))    #构建损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5) #构建加速方法
train = optimizer.minimize(loss)        #将加速方法和损失函数定义一起

# init = tf.initialize_all_variables() #2017年3月后用 global_variables_initializer()代替。
init = tf.global_variables_initializer() #注意别漏了这句！
### create tensorflow structure end ###


sess = tf.Session()
sess.run(init)  # Very important

# train
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

# 运行结果：
'''
0 [-0.35102686] [ 0.7827968]
20 [-0.07313562] [ 0.39562976]
40 [ 0.04549079] [ 0.33010763]
60 [ 0.08283859] [ 0.30947897]
80 [ 0.09459697] [ 0.30298433]
100 [ 0.09829891] [ 0.30093959]
120 [ 0.09946444] [ 0.30029583]
140 [ 0.09983138] [ 0.30009314]
160 [ 0.09994691] [ 0.30002934]
180 [ 0.09998328] [ 0.30000925]
200 [ 0.09999475] [ 0.3000029]

Process finished with exit code 0
'''
