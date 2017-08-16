import numpy as np
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

print(mnist.train.next_batch(1))