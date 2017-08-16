import tensorflow as tf

# 限制显卡显存
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 常量算法
a = tf.constant(1)
b = tf.constant(2)

with tf.Session() as sess:
    print(sess.run(a + b))
    print(sess.run(a * b))

# 定义方法
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print(sess.run(add))
    print(sess.run(mul))

# 定义输入
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

add = tf.add(input1, input2)
mul = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(add, feed_dict={input1: [4], input2: [5]}))
    print(sess.run(mul, feed_dict={input1: [4], input2: [5]}))

# 矩阵乘法
matrix1 = tf.constant([[2, 3]])
matrix2 = tf.constant([[2],
                       [3]])

mul = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    print(sess.run(mul))
