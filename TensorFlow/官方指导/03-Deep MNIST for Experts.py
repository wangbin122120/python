import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# define weight and bias 因为后面的激励函数用的是Relu，如果W从0开始，有可能会变成“dead neurons"
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define layel:Convolution and Pooling  'SAME'的目的是保持输入和输出的shape一致，这样可以避免一些shape上的计算
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    # pool用的是max_pool，是选取blocks 2x2（[1,2,2,1]）中4个pixs的最大一个pixs值。 ksize是blocks（窗口）大小，strides移动步伐
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# define cnn ,Build a Multilayer Convolutional Network
def deepnn(x):
    # reshape
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # layel1: First convolutional layer - maps one grayscale image to 32 feature maps.
    # 其实这一层的计算还是和之前的很相同。都是定义w,b再相乘，只不过这里的W,b以及用的乘法和激励函数复杂很多。
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])  #@ W 块（ patch size）大小是5x5这点非常容易忘记，通过参数训练而得，后面两个是：输入个数，输出个数
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        # @这里将keep_prob参数化的原因是：训练的时候0.5,测试时候为1.0；
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


# 以下和前面的softmax mnist 没有太大区别
# Import data
mnist = input_data.read_data_sets('/home/w/tmp/tensorflow/mnist/input_data', one_hot=True)

# define model,Build the graph for the deep net
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
logits, keep_prob = deepnn(x)

# define loss and train
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
cross_entropy = tf.reduce_mean(cross_entropy)  # 这里拆开了

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    #@这里不用梯度下降，梯度下降很不稳定，且结果不好，区别很大。

# define test model
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# define summary
graph_location = '03_mnist_deep_summary'
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

# running
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # training
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # 这里的eval等同于sess.run(accuracy,)
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %0.4f' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    # test model 因为笔记本的内存容量限制，不能全部装载，所以拆成50份测试，每次测试200个，取最后的平均值。
    acc = 0.0
    for i in range(50):
        acc += accuracy.eval(feed_dict={
            x: mnist.test.images[i * 200:(i + 1) * 200, ], y: mnist.test.labels[i * 200:(i + 1) * 200, ],
            keep_prob: 1.0})
    print('test accuracy %0.4f' % (acc / 50.0))


'''运行结果：
step 2700, training accuracy 0.9800
step 2800, training accuracy 1.0000
step 2900, training accuracy 0.9600
test accuracy 0.9812

Process finished with exit code 0

########################### shape变化情况：
input       N,784
    reshape N,28,28,1

conv1       N,28,28,32
    W       5,5,1,32
    b       32
    --> tf.nn.relu( b + conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

pool1       N,14,14,32
    --> tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
conv2       N,14,14,64
    W       5,5,32,64
    b       64
    --> tf.nn.relu( b + conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))    

pool2       N, 7, 7,64
    --> tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

fc1         N,1024        
    reshape N, 7*7*64
       W    7*7*64 , 1024
       b    1024
    --> tf.nn.relu(reshape*W+b) 
    
dropout     N,1024   

fc2         N,10
       W    N,1024,10
       b    N,10
    --> x*W +b
    

'''