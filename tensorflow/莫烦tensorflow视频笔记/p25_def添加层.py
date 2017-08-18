import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)


# 定义一个
def add_layer(inputs, in_size, out_size, activation_function=None, layer_name='layer'):
    # W和b在初始时为0是很不好的，随机生成。
    # 在p29课中添加name_scope:
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), dtype=tf.float32, name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # @注意此句不等于： Wx_plus_b = tf.add(tf.multiply(Weights, inputs), biases)

        # 如果没有定义激励函数，那么就用初始的线性函数，否则多加一层则增加一个非线性函数
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)  # 这里不需要处理名字，因为每个激励函数自身都有
        return outputs
