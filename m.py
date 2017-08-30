import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from IPython.display import Image

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 提取图片数据
mnist = input_data.read_data_sets("/home/w/tmp/tensorflow/mnist/input_data", one_hot=True)


# define weights and bias
def weights_variable(shape):
    return tf.Variable(tf.truncated_normal(dtype=tf.float32, stddev=0.1, shape=shape))


def bias_variabel(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# define layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define cnn
def deep_cnn(x):
    # reshape       x[N,28*28] --> x_img[N,28,28,1]
    with tf.name_scope('reshape'):
        x_img = tf.reshape(x, shape=[-1, 28, 28, 1])

    # L1 conv1      x_img[N,28,28,1] --> c1[N,28,28,32]
    with tf.name_scope('reshape'):
        W1 = weights_variable([5, 5, 1, 32])
        b1 = bias_variabel([32])
        L1 = tf.nn.relu(conv2d(x_img, W1) + b1)

    # L2 pool1      c1[N,28,28,32] --> p1[N,14,14,32]
    with tf.name_scope('L2_pool1'):
        L2 = max_pool_2x2(L1)

    # L3 conv2      p1[N,14,14,32] --> c2[N,14,14,64]
    with tf.name_scope('L3_conv2'):
        W3 = weights_variable([5, 5, 32, 64])
        b3 = bias_variabel([64])
        L3 = tf.nn.relu(conv2d(L2, W3) + b3)

    # L4 pool2      c2[N,14,14,64] --> p2[N,7,7,64]
    with tf.name_scope('L4_pool2'):
        L4 = max_pool_2x2(L3)

    # L5 func1      p2[N,7,7,64] --> f1[N,1024]
    with tf.name_scope('L5_func1'):
        L4_flat = tf.reshape(L4, shape=[-1, 7 * 7 * 64])
        W5 = weights_variable([7 * 7 * 64, 1024])
        b5 = bias_variabel([1024])
        L5 = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)

    # L6 dropout    f1[N,1024] --> d1[N,1024]
    with tf.name_scope('L6_dropout'):
        keep_prob = tf.placeholder(tf.float32)
        L6 = tf.nn.dropout(L5, keep_prob)

    # L7 func2      d1[N,1024] --> f2[N,10]
    with tf.name_scope('L7_func2'):
        W7 = weights_variable([1024, 10])
        b7 = bias_variabel([10])
        L7 = tf.matmul(L6, W7) + b7

    logits = L7
    return logits, keep_prob


# define model
x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])
logits, keep_prob = deep_cnn(x)

# define loss and optimizer
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# define test
with tf.name_scope('test'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))

# define summary
graph_location = '03_mnist_deep_summary'
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

# running
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # training
    for step in range(1000):
        xtr, ytr = mnist.train.next_batch(50)
        if step % 100 == 0:
            print('step:', step, ' acc:', sess.run(accuracy, {x: xtr, y: ytr, keep_prob: 1.0}))
        sess.run(train_step, {x: xtr, y: ytr, keep_prob: 0.5})

    # test
    xte, yte = mnist.test.next_batch(500)
    print('real accuracy:', sess.run(accuracy, {x: xte, y: yte, keep_prob: 1.0}))
