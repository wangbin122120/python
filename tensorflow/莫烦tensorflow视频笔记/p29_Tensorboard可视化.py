import matplotlib.pylab as plt

# 是《p26_建造神经网络》程序的例子
# 在 《p27_结果可视化》添加了plt的图形输出
'''
本次程序，添加tensorboard的可视化输出，具体浏览器中的查看方法如下：
比如最终Summry写到目录“D:\wangbin\project\python\git\tensorflow\莫烦tensorflow视频笔记\p29”，那在终端执行：
C:\Users\wangbin>d:
D:\>cd D:\wangbin\project\python\git\tensorflow\莫烦tensorflow视频笔记\
D:\wangbin\project\python\git\tensorflow\莫烦tensorflow视频笔记>tensorboard --logdir='p29/'
Starting TensorBoard b'47' at http://0.0.0.0:6006
(Press CTRL+C to quit)
接着在浏览器中打开 网址：localhost:6006

'''
import numpy as np
import tensorflow as tf

# 限制显卡内存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

from p25_def添加层 import add_layer

# 设定输入数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # ，在最后一个维度后增加维度。
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# # 建立典型的三层神经网络:输入层-隐藏层-输出层

# 设定参数，要明白设置placeholder，是为了提升训练的效率,因为不需要用到所有数据
# xs = tf.placeholder(tf.float32, [None, 1])
# ys = tf.placeholder(tf.float32, [None, 1])
with tf.name_scope('inputs'):  # tensorboard 中命名框
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # 定义name=数据节点名称
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# @神经网络最关键的两步：
# 设置隐藏层，输入个数是由xs决定，个数是1，设定给10个神经元，其激励函数用relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu, layer_name='hidden_layer')
# 设置输出层，输入个数是由上一层l1决定，个数是10，输出结果只有一个，输出层不需要设定激励函数
predition = add_layer(l1, 10, 1, activation_function=None, layer_name='output_layer')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                                    reduction_indices=1))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


# 可视化过程：
# 这段程序最好放在外侧，
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()  # 有了这句话，在画图的过程中不会中断程序，不需要人工干预就自动接着执行。 对应的关闭方法：plt.ioff()
plt.show()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('p29/',sess.graph)  #@非常关键的一步：将结果加载到文件中，之后才能在浏览器中查看，并且要在Session()定义之后，否则没有任何对象。
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            loss_value = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            predition_value = sess.run(predition, feed_dict={xs: x_data})

            # 因为循环过程会不断产生lines,在画新lines时，应该先将旧lines删除，由于第一次删除时没有线条，所以会报错，跳过即可。这样最后程序结束时正好保留最终的预测曲线。
            try:
                ax.lines.remove(lines[0])
                # @不要错写成：fig.remove(lines[0])，这样删不掉的。
            except Exception:
                pass
            plt.xlabel('loss=%.5f' % loss_value)
            lines = ax.plot(x_data, predition_value, 'r-', lw=5)  # lw线条宽度。在ax图片上画预测曲线，因为后面要清除，所以需要一个返回值lines。
            plt.pause(0.1)
