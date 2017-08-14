# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.

使用方法：
1.运行本程序，等待结束
2.在cmd终端中运行：
  d:
  cd D:\Anaconda3\Scripts\
  tensorboard.exe --logdir=D:\tmp\tensorflow\mnist\logs\mnist_with_summaries
  其中可以指定端口和IP
  tensorboard.exe --host 127.0.0.2 --logdir=D:\tmp\tensorflow\mnist\logs\mnist_with_summaries
3.在浏览器中打开：
  localhost:6006
  或者
  http://127.0.0.1:6006/
4.如果需要重新运行程序，需要将cmd后台退出。


代码讲解常见：
http://blog.csdn.net/sinat_33761963/article/details/62433234


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None



def train():
  '''
    # Import data
  （3）接着加载数据,下载数据是直接调用了tensorflow提供的函数read_data_sets,
  输入两个参数，第一个是下载到数据存储的路径，
  第二个one_hot表示是否要将类别标签进行独热编码。
  它首先回去找制定目录下有没有这个数据文件，没有的话才去下载，有的话就直接读取。
  所以第一次执行这个命令，速度会比较慢。
  '''
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
  '''
  2.2 创建特征与标签的占位符，保存输入的图片数据到summary
  （1）创建tensorflow的默认会话：
  '''
  sess = tf.InteractiveSession()
  # Create a multilayer model.

  '''
  （2）创建输入数据的占位符，分别创建特征数据x，标签数据y_ 
  在tf.placeholder()函数中传入了3个参数，
  第一个是定义数据类型为float32；
  第二个是数据的大小，特征数据是大小784的向量，标签数据是大小为10的向量，
      None表示不定死大小，到时候可以传入任何数量的样本；
  第3个参数是这个占位符的名称。
  '''
  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  '''
  （3）使用tf.summary.image保存图像信息 
特征数据其实就是图像的像素数据拉升成一个1*784的向量，
现在如果想在tensorboard上还原出输入的特征数据对应的图片，
就需要将拉升的向量转变成28 * 28 * 1的原始像素了，
于是可以用tf.reshape()直接重新调整特征数据的维度： 
将输入的数据转换成[28 * 28 * 1]的shape，存储成另一个tensor，命名为image_shaped_input。 
为了能使图片在tensorbord上展示出来，使用tf.summary.image将图片数据汇总给tensorbord。 
tf.summary.image（）中传入的第一个参数是命名，第二个是图片数据，第三个是最多展示的张数，此处为10张
  '''
  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


  '''
  2.3 创建初始化参数的方法，与参数信息汇总到summary的方法
  （1）在构建神经网络模型中，每一层中都需要去初始化参数w,b,
  为了使代码简介美观，最好将初始化参数的过程封装成方法function。 
创建初始化权重w的方法，生成大小等于传入的shape参数，标准差为0.1,正态分布的随机数，
并且将它转换成tensorflow中的variable返回。
  '''
  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#创建初始换偏执项b的方法，生成大小为传入参数shape的常数0.1，并将其转换成tensorflow的variable并返回
  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  '''
  （2）我们知道，在训练的过程在参数是不断地在改变和优化的，
  我们往往想知道每次迭代后参数都做了哪些变化，可以将参数的信息展现在tenorbord上，
  因此我们专门写一个方法来收录每次的参数信息。
      使用tf.summary.scalar 记录标量 
      使用tf.summary.histogram 记录数据的直方图 
      使用tf.summary.distribution 记录数据的分布图 
      使用tf.summary.image 记录图像数据 
  '''
  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


  '''
  2.4 构建神经网络层
  （1）创建第一层隐藏层 
    创建一个构建隐藏层的方法,输入的参数有： 
    input_tensor：特征数据 
    input_dim：输入数据的维度大小 
    output_dim：输出数据的维度大小(=隐层神经元个数） 
    layer_name：命名空间 
    act=tf.nn.relu：激活函数（默认是relu)
  '''
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    # 设置命名空间
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      # 执行wx+b的线性计算，并且用直方图记录下来
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      # 将线性输出经过激励函数，并将输出也用直方图记录下来
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)

      # 返回激励层的最终输出
      return activations

  #调用隐层创建函数创建一个隐藏层：输入的维度是特征的维度784，神经元个数是500，也就是输出的维度。
  hidden1 = nn_layer(x, 784, 500, 'layer1')

  #（2）创建一个dropout层，,随机关闭掉hidden1的一些神经元，并记录keep_prob
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  #（3）创建一个输出层，输入的维度是上一层的输出: 500, 输出的维度是分类的类别种类：10
  #，激活函数设置为全等映射identity.  （暂且先别使用softmax, 会放在之后的损失函数中一起计算）
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)


  '''
  2.5 创建损失函数

  使用tf.nn.softmax_cross_entropy_with_logits来计算softmax并计算交叉熵损失,
  并且求均值作为最终的损失值。
  '''
  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  '''
  2.6 训练，并计算准确率
  （1）使用AdamOptimizer优化器训练模型，最小化交叉熵损失
  '''
  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  #（2）计算准确率,并用tf.summary.scalar记录准确率
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      # 求均值即为准确率
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  '''
  2.7 合并summary operation, 运行初始化变量

  将所有的summaries合并，并且将它们写到之前定义的log_dir路径
  '''
  merged = tf.summary.merge_all()
  # 写到指定的磁盘路径中
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  ## 运行初始化所有变量
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries
  '''
  2.8 准备训练与测试的两个数据，循环执行整个graph进行训练与评估

（1）现在我们要获取之后要喂人的数据. 
如果是train==true，就从mnist.train中获取一个batch样本，并且设置dropout值； 
如果是不是train==false,则获取minist.test的测试数据，并且设置keep_prob为1，即保留所有神经元开启
  '''
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  '''（2）开始训练模型。 
每隔10步，就进行一次merge, 并打印一次测试数据集的准确率，
然后将测试数据集的各种summary信息写进日志中。 
每隔100步，记录原信息 
其他每一步时都记录下训练集的summary信息并写到日志中。
  '''
  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      # 记录测试集的summary与accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      # 记录训练集的summary
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


'''
2.Tensorboard使用案例
（2）定义固定的超参数,方便待使用时直接传入。
如果你问，这个超参数为啥要这样设定，如何选择最优的超参数？
这个问题此处先不讨论，超参数的选择在机器学习建模中最常用的方法就是“交叉验证法”。
而现在假设我们已经获得了最优的超参数，
设置学利率为0.001，dropout的保留节点比例为0.9，最大循环次数为1000.

max_step = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例

另外，还要设置两个路径，第一个是数据下载下来存放的地方，
一个是summary输出保存的地方。

data_dir = ''   # 样本数据存储的路径
log_dir = ''    # 输出日志保存的路径
 
'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt output.txt --user=name --port=8080。
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')

  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')

  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

