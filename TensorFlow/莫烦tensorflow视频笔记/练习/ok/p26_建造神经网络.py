import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

import numpy as np
from p25_def添加层 import add_layer

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) + 2 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 隐藏层：
l1 = add_layer(xs, 1, 10, tf.nn.relu)
# 输出层：
pred = add_layer(l1, 10, 1, None)

# 定义损失函数和优化方法
loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - ys), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        sess.run(optimizer, feed_dict={xs: x_data, ys: y_data})
        if i % 500 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    print('end')
