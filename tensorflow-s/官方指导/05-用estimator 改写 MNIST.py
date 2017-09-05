## 简单的单层dnn网络

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

# Read in the MNIST dataset
mnist = input_data.read_data_sets("/home/w/tmp/tensorflow/mnist/input_data", one_hot=False)

# define estimator
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=[tf.contrib.layers.real_valued_column('', dimension=28 * 28)],
    hidden_units=[1024],
    n_classes=10,
    optimizer=tf.train.AdamOptimizer())

# training
classifier.fit(x=mnist.train.images,
               y=mnist.train.labels.astype(np.int32),
               batch_size=100,
               steps=1000)

# test
eval_result = classifier.evaluate(x=mnist.test.images,
                                  y=mnist.test.labels.astype(np.int32),
                                  steps=1)

print('accuracy:', eval_result['accuracy'])

#######################################################################

## 复杂的cnn网络
## https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/layers/cnn_mnist.py
import numpy as np
import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    def conv_layer(inputs, filters, kernel_size=[5, 5], padding='same', activation=tf.nn.relu):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation)

    def pool_layer(inputs,
                   pool_size=[2, 2],
                   strides=2):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=pool_size,
            strides=strides)

    # input
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # conv 1
    conv1 = conv_layer(input_layer, 32)

    # pooling layer 1
    pool1 = pool_layer(conv1)

    # conv 2
    conv2 = conv_layer(pool1, 64)

    # pooling layer 2
    pool2 = pool_layer(conv2)

    # func 1
    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # drop out
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # output
    logits = tf.layers.dense(inputs=dropout, units=10)

    # 以上是普通cnn结构，但用estimator，还需要定义 loss，optimizer,prediction
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss(both train,eval) @这个定义还必须在 predict 之后！否则会出错，感觉很诡异啊！！！
    loss = tf.losses.softmax_cross_entropy(logits=logits,
                                           onehot_labels=tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10))
    #
    if mode == tf.estimator.ModeKeys.TRAIN:  # optimizer(only for train)
        op = tf.train.GradientDescentOptimizer(0.001).minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metric_op = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_op)


# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/home/w/tmp/tensorflow/mnist_convnet_model")

# Set up logging for predictions
logging_hook = tf.train.LoggingTensorHook(tensors={"probabilities": "softmax_tensor"}, every_n_iter=500)

#  train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
classifier.train(input_fn=train_input_fn,
                 steps=500,
                 hooks=[logging_hook])

# evaluate model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_v = classifier.evaluate(input_fn=eval_input_fn)
print('evaluate:', eval_v)

# prediction test data
prediction_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': eval_data[0:10]},
    num_epochs=1,
    shuffle=False)
predic_result = classifier.predict(input_fn=prediction_input_fn)
predic_class = [x['classes'] for x in predic_result]

for i in range(10):
    print('预测结果：', predic_class[i], ' 真实结果：', eval_labels[i])
