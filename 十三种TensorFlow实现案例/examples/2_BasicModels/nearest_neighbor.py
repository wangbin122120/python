import numpy as np
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200)  # 200 for testing
# 注意，Ytr和Yte的每个标签都是如下格式：[ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
# 如需知道具体是数字几，还需要用argmax()求得实际位置。
# Xtr.shape=(5000, 784)
# 注意这里，已经如同一般的模型对图片进行降维处理，不需要重复处理。

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance


distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 1.关于negative
# tf.add(xtr, tf.negative(xte)) 用了negative就是取负数，xtr + (-xte) 相当于 xtr - xte

# 2.关于reduction_indices
# reduce_sum() 中的 reduction_indices 相当于 axis 的作用，不清楚两者的差别。
# 对1 计算，则就是对每张图片进行计算，分别求出距离，所以结果是 （len(xtr),1）
# 如果改成 0 计算，则是所有图片之和，那就只有一个值 （1）那没有意义，根本区分不出和哪个图片距离最近。

# 3.后续如何调用
# 后面调用的时候用的是：
# pred = tf.arg_min(distance, 0)
# nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})

# 4.关于其中的xtr xte变量
# 涉及公式中的变量，往往都需要预先定义，就是上面的xtr,xte，虽然知道是要传什么，但不建议直接将数据写死。
# 写死后如下，
# distance = tf.reduce_sum(tf.abs(tf.add(Xtr, tf.negative(xte))), reduction_indices=1)
# 调用的时候是：nn_index = sess.run(pred, feed_dict={xte: Xte[i, :]})
# 结果准确率变低了：由0.92降低为：Accuracy: 0.8800000000000007

# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)
# 对distance数组中求出最小元素的位置。相当于np.argmin()

accuracy = 0.
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor，对每个图片都寻找最小距离的位置
        # nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i]})  # 一样的，这样更好理解
        # nn_index = sess.run(pred, feed_dict={xte: Xte[i, :]})

        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), Ytr[nn_index],
              "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
