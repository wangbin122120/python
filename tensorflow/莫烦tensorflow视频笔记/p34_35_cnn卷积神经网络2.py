# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 谷歌自己也用normal产生数据
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # bias通常用正数效果较好
    return tf.Variable(initial)


# 定义卷积神经网络层
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # 2d是二维。
    # x-数据,W
    # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    #       [1,x_move,y_move,1]，前后两个1是固定不变。中间两个x_move，x轴行向量pix个数，y_move是y轴列向量pix个数
    # padding有两种，valid和SAME，SAME是保持卷积前后的shape大小不变。


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # pool是承接在conv2d()之后，将数据shape减少，不需要W，
    # ksize：池化窗口的大小，[1, height, width, 1]
    # strides是[1,x_move,y_move,1],扫描步数，这点和conv2d是一样的。


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # 将图片信息还原。
#接下来是用卷积神经网络，所以要变换shape,立体还原；这点和普通神经网络的输入情况（扁平化）是不同的。
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])
# 一次扫描的宽度：patch 5x5, 厚度：in size 1；优化后得到的输出高度:output depth 32
b_conv1 = bias_variable([32])  # 跟随输出高度
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# @注意使用非线性化relu()，@注意别忘了加b_conv1
# 之前是x*W，现在用conv2d(x,W)来代替直接相乘。 output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
# @注意非线性化relu()
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## func1 layer ##
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) #@这里输入值是h_pool2，别弄错了。程序太容易写错了
#接下来是用普通神经网络，所以要重新变换shape,扁平化；这点和卷积神经网络的输入情况是不同的。
#配合着将W,b的shape也做相应的调整 ，1024应该是交叉验证的结果是1024结果比较理想。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#第一次一般都是用relu进行非线性化处理。
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#最后的输出结果用softmax计算概率。输出值为10个分类结果。
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#这次不用梯度下降的原因是系统复杂，计算量大。
#AdamOptimizer

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
