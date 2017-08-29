# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('/home/w/tmp/tensorflow/mnist/input_data/', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 10000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32,[None,n_inputs,n_steps])
y = tf.placeholder(tf.float32,[None,n_classes])
# Define weights
weights={

    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}



def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X,[-1,n_inputs])

    # into hidden
    # X_in =
    X_in = tf.matmul(X,weights['in'])+biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell =
    else:
        cell =
    # lstm cell is divided into two parts (c_state, h_state)
    init_state =

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state =

    # hidden layer for output as the final results
    #############################################
    # results =

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs =
    else:
        outputs =
    results =

    return results


pred =
cost =
train_op =

correct_pred =
accuracy =

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init =
    else:
        init =
    sess.run(init)
    step =
    while step * batch_size < training_iters:
        batch_xs, batch_ys =
        batch_xs =
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1



