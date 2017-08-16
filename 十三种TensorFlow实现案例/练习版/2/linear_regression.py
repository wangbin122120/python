import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 参数设定
learning_rate = 0.1
traing_epotchs = 100
display_step = 20

# 训练数据
# train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# 用自建测试数据
train_X = 2 * np.random.randn(20) + 3
train_Y = 3 * train_X + 2 + np.random.rand()
n_samples = train_X.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 设置训练模型
# W = tf.Variable(tf.random_uniform([1]), name='weights')
# b = tf.Variable(tf.random_uniform([1]), name='bias')
rng = np.random
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# 建立线性模型
pred = tf.add(tf.multiply(W, train_X), b)

# 建立损失函数
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    plt.figure(figsize=(60, 20))
    p = []
    for i in range(int(traing_epotchs / display_step)):
        p.append(plt.subplot(321 + i))
    print('\n', len(p))

    for step in range(traing_epotchs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        if (step + 1) % (display_step) == 0:
            i = int(step / display_step)
            print(i)
            p[i].plot(train_X, train_Y, 'ro', label='Original data')
            p[i].plot(train_X, sess.run(W) * train_X + sess.run(b), label='W=%s,b=%s' % (sess.run(W), sess.run(b)))
            p[i].set_ylabel('cost=%s' % sess.run(cost, feed_dict={X: train_X, Y: train_Y}), fontsize=7)
            p[i].legend()
    plt.show()

    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    # 测试数据
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray(3 * test_X + 2)

    print("Testing... (Mean square loss Comparison)")
    pred = tf.add(tf.multiply(W, test_X), b)
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above

    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(test_X, sess.run(W) * test_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
