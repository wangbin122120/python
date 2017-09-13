from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

'''
这个程序，说明dropout 避免过拟合的作用，就是在每次训练的过程中，都会去掉一部分测试用例，以防止过拟合。
在程序隐藏层和结果层进行限制结果比例，比如：
    输出结果=tf.nn.dropout(输出结果,keep_prob=保留比例)
    在《p25_def添加层》：
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
然后在调用时，将keep_prob传入：
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})

但是很尴尬的是，无论如何调整，在最终的结果都没有出现过拟合的情况，不知道算法是否都已经优化，所以这个作用并不显著。    
'''
# 限制显卡显存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

# load data
digits = load_digits()
X = digits.data  # 添加0-9数字的图片，有[1797][8x8]
y = digits.target  # [1797] 标签列表 [0 1 2 ..., 8 9 8],
y = LabelBinarizer().fit_transform(y)  # [1797][10]类似mnist中的标签表示方法，对应位置取1，其他为0的10个标记位。

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

from p25_def添加层 import add_layer

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 100, activation_function=tf.nn.tanh, layer_name='l1',keep_prob=keep_prob)  # 除了tf.nn.tanh ，其他有些问题。
prediction = add_layer(l1, 100, 10, activation_function=tf.nn.softmax, layer_name='l2',keep_prob=keep_prob)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("/tmp/tensorflow/mofan_logs/p32/train", sess.graph)
test_writer = tf.summary.FileWriter("/tmp/tensorflow/mofan_logs/p32/test", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
    # 训练的时候，dropout一半，避免过饱和。
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
