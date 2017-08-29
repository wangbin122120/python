import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# define training data
x_real = [1, 2, 3, 4, 5]
y_real = [0, 1, 2, 3, 4]

# define placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# define prediction model
W = tf.Variable(tf.zeros([1]), tf.float32)
b = tf.Variable(tf.zeros([1]), tf.float32)
linear_model = W * x + b

# define loss model
loss = tf.reduce_sum(tf.square(linear_model - y))

# define train model
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# running all
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # trainning
    for _ in range(1000):
        sess.run(train, {x: x_real, y: y_real})
    print(sess.run(loss, {x: x_real, y: y_real}))
    print(sess.run([W, b]))
