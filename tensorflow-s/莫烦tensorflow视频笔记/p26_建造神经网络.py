import numpy as np
import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

from p25_def添加层 import add_layer

# 设定输入数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # ，在最后一个维度后增加维度。
'''
1。生成序列中，newaxis的用法，增加一个维度
print(np.linspace(-1, 1, 300).shape) # linspace仅仅产生一个一维列表 (300,)
print(x_data.shape)  #通过 (300, 1)
    生成结果：
    [[-1.        ]
     [-0.99331104]
     [-0.98662207]
     ...
              [1]]
对比：np.linspace(-1, 1, 300)[np.newaxis].shape=(1, 300)，在最前面增加维度
对比: np.linspace(-1, 1, 300)[:, np.newaxis][:, np.newaxis].shape=(300, 1, 1)
    [[[-1.        ]]
     [[-0.99331104]]
     [[-0.98662207]]
'''

noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# # 建立典型的三层神经网络:输入层-隐藏层-输出层

# 设定参数，要明白设置placeholder，是为了提升训练的效率,因为不需要用到所有数据
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# @神经网络最关键的两步：
# 设置隐藏层，输入个数是由xs决定，个数是1，设定给10个神经元，其激励函数用relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 设置输出层，输入个数是由上一层l1决定，个数是10，输出结果只有一个，输出层不需要设定激励函数
predition = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                                    reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

'''运行结果
0.630816
0.0207086
0.0114395
0.00788576
0.00633314
0.00560672
0.00506055
0.00467311
0.00441625
0.00422962
0.00409234
0.00396903
0.00386666
0.00377447
0.00368991
0.003613
0.00353603
0.00347075
0.00341302
0.00335801
'''
