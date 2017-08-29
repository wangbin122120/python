import tensorflow as tf
# 限制显卡显存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True #开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

Xtr,Ytr=mnist.train.next_batch(5000)
Xte,Yte=mnist.test.next_batch(500)

from p25_def添加层 import add_layer

with tf.name_scope('input_layer'):
    xs=tf.placeholder(tf.float32,[None,28*28],name='x_input')
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')

with tf.name_scope('output_layer'):
    prediction=add_layer(xs,784,10,tf.nn.softmax,'output')

with tf.name_scope('loss'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                                reduction_indices=1),name='loss')
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    def accuracy_calc(x, y):
        global prediction
        pred_y = sess.run(prediction, feed_dict={xs: x})
        is_correct = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        return sess.run(accuracy)

    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('/tmp/tensorflow/mofan_logs/p31/',sess.graph)
    for i in range(1000):
        sess.run(train_step,feed_dict={ys:Ytr,xs:Xtr})
        if (i+1) % 50 ==0:
            print('#',i,' accuracy=',accuracy_calc(Xte,Yte),' loss=',sess.run(cross_entropy,feed_dict={ys:Ytr,xs:Xtr}))