# 之前所有问题都是回归问题，适用于连续分布，比如价格预测
# calssfications是分类问题，很经典的数字识别
import tensorflow as tf
# 限制显卡显存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True #开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200)  # 200 for testing

from p25_def添加层 import add_layer

#定义placeholder
with tf.name_scope('input_layer'):
    xs = tf.placeholder("float", [None, 784],name='img') #28x28
    ys = tf.placeholder("float", [None, 10],name='label') #

with tf.name_scope('output_layer'):
    prediction = add_layer(xs,784,10,tf.nn.softmax,'output')

with tf.name_scope('loss'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                                reduction_indices=1),name='loss')

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:

    #计算准确率(0~1)
    def accuracy_calc(x, y):
        global prediction
        pred_y = sess.run(prediction, feed_dict={xs: x})
        is_correct = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
        # print(tf.cast(is_correct,tf.float32))
        # 由于tf.equal返回的是bool类型，不方便用tf.reduce_mean()直接进行统计，所以做一次cast()变成0,1就方便了。
        accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
        #注意，tf 返回的是一个tensor，直接打印是没有任何意义的，需要进行一次run()，返回值也是同理。
        return sess.run(accuracy)

    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('/tmp/tensorflow/mofan_logs/p31/',sess.graph)
    for i in range(1000):
        sess.run(train_step,feed_dict={ys:Ytr,xs:Xtr})
        if (i+1) % 50 ==0:
            print('#',i,' accuracy=',accuracy_calc(Xte,Yte),' loss=',sess.run(cross_entropy,feed_dict={ys:Ytr,xs:Xtr}))

'''
# 49  accuracy= 0.62  loss= 1.95412
# 99  accuracy= 0.72  loss= 1.28385
# 149  accuracy= 0.75  loss= 1.03434
# 199  accuracy= 0.785  loss= 0.89159
# 249  accuracy= 0.805  loss= 0.794482
# 299  accuracy= 0.815  loss= 0.722399
# 349  accuracy= 0.83  loss= 0.665955
# 399  accuracy= 0.83  loss= 0.620126
# 449  accuracy= 0.84  loss= 0.581932
# 499  accuracy= 0.855  loss= 0.549422
# 549  accuracy= 0.855  loss= 0.521264
# 599  accuracy= 0.855  loss= 0.496543
# 649  accuracy= 0.855  loss= 0.474615
# 699  accuracy= 0.855  loss= 0.455011
# 749  accuracy= 0.86  loss= 0.437371
# 799  accuracy= 0.865  loss= 0.421408
# 849  accuracy= 0.865  loss= 0.406887
# 899  accuracy= 0.87  loss= 0.393615
# 949  accuracy= 0.87  loss= 0.381428
# 999  accuracy= 0.875  loss= 0.370186

Process finished with exit code 0
'''