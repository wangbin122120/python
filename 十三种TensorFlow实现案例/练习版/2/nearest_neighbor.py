import numpy as np
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

Xtr,Ytr=mnist.train.next_batch(5000)
Xte,Yte=mnist.train.next_batch(500)

xtr=tf.placeholder(tf.float32,[None,784])
xte=tf.placeholder(tf.float32,[784])

dis=tf.arg_min(tf.reduce_sum(tf.square(xtr-xte),reduction_indices=1),0)

init=tf.global_variables_initializer()

acc=0
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xte)):
        index=sess.run(dis,feed_dict={xtr:Xtr,xte:Xte[i]})
        # print(index)
        # print(np.argmax(Ytr[index]),np.argmax(Yte[i]))
        if np.argmax(Ytr[index])==np.argmax(Yte[i]):
            acc+=1
    print('准确率',acc/len(Xte))

