'''The example demonstrates how to write custom layers for Keras.

We build a custom activation layer called 'Antirectifier',
which modifies the shape of the tensor that passes through it.
We need to specify two methods: `compute_output_shape` and `call`.

Note that the same result can also be achieved via a Lambda layer.

Because our custom layer is written with primitives from the Keras
backend (`K`), our code can run both on TensorFlow and Theano.
该示例演示了如何为Keras编写自定义图层。

我们构建一个名为“Antirectifier”的定制激活层，
这改变了通过它的张量的形状。
我们需要指定两个方法：`compute_output_shape`和`call`。

注意，也可以通过Lambda层实现相同的结果。

因为我们的自定义层是用Keras的原语写的
后端（`K`），我们的代码可以在TensorFlow和Theano上运行。
'''

from __future__ import print_function

# 如果是keras，前面加上这两句
import tensorflow as tf

import keras
from keras import Sequential
from keras import backend as K
from keras import layers
from keras import mnist

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True #开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)


class Antirectifier(layers.Layer):
    '''This is the combination of a sample-wise
    L2 normalization with the concatenation of the
    positive part of the input with the negative part
    of the input. The result is a tensor of samples that are
    twice as large as the input samples.

    It can be used in place of a ReLU.

    # Input shape
        2D tensor of shape (samples, n)

    # Output shape
        2D tensor of shape (samples, 2*n)

    # Theoretical justification
        When applying ReLU, assuming that the distribution
        of the previous output is approximately centered around 0.,
        you are discarding half of your input. This is inefficient.

        Antirectifier allows to return all-positive outputs like ReLU,
        without discarding any data.

        Tests on MNIST show that Antirectifier allows to train networks
        with twice less parameters yet with comparable
        classification accuracy as an equivalent ReLU-based network.
这是一个样本的组合
     L2标准化与串联
     正面部分的输入与负部分
     的输入。 结果是一个样本的张量
     是输入样本的两倍。

     它可以用来代替ReLU。

     ＃输入形状
         二维形状张量（样本，n）

     ＃输出形状
         2D张量的形状（样本，2 * n）

     理论理由
         当应用ReLU时，假设分配
         以前的输出约为0.左右，
         你正在丢弃一半的输入。 这是无效的。

         控制器允许返回所有的正输出，如ReLU，
         而不会丢弃任何数据。

         对MNIST的测试表明，Antirectifier允许训练网络
         具有两倍的参数与可比较
         分类精度作为等价的基于ReLU的网络。
    '''

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)

# global parameters
batch_size = 128
num_classes = 10
epochs = 40

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices ，转换数据结构，原本 y_train 是列表，存储 60000个 0-9的数字，经过转换，变成 (60000,10)的元组，10个0或1 表示对应位置的数字。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build the model 将一些网络层通过.add()堆叠起来，就构成了一个模型：
'''
顺序模型的最基本步骤：
model = Sequential()  #按顺序的
model.add(Dense(units=64, input_dim=100)) #dense:全连接的神经层，units 是输出个数
model.add(Activation("relu"))
model.add(Dense(units=10))  # 第二层的默认input_dim 等于第一层的 units 
model.add(Activation("softmax"))
'''
model = Sequential()
model.add(layers.Dense(256, input_shape=(784,)))
model.add(Antirectifier()) # 自定义的 Activation()
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

# compile the model 完成模型的搭建后，我们需要使用.compile()方法来编译模型：
# 编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。
model.compile(loss='categorical_crossentropy', # 损失函数，比如 MSE
              optimizer='rmsprop',             # 优化器，比如 sgd
              metrics=['accuracy']) # 在优化的同时也计算误差或准确率等。

IsSingle = True
if IsSingle is True:
    model.train_on_batch(x_train,y_train) # 单步训练
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)  # 输出loss值和准确性
    classes = model.predict(x_test, batch_size=128) # 得到预测结果，是一个（10000,10）元组，第二位的0~9号存储的是对应结果的“概率值”。
else:
    # train the model 完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

# next, compare with an equivalent network
# with2x bigger Dense layers and ReLU
