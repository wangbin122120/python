import tensorflow as tf

# 限制显卡显存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True  # 开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)
#
# # 保存数据
# W = tf.Variable([[1, 2, 3], [2, 3, 4]], dtype=tf.float32)
# b = tf.Variable([1, 2, 3], dtype=tf.float32)
#
# save = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print('path:', save.save(sess, '/tmp/tensorflow/mofan_logs/p36/save2.ckpt'))


# 提取数据
import numpy as np
W = tf.Variable([[2, 2, 1], [2, 3, 4]], dtype=tf.float32)
b = tf.Variable([2, 2, 1], dtype=tf.float32)

save=tf.train.Saver()
with tf.Session() as sess:
    save.restore(sess,'/tmp/tensorflow/mofan_logs/p36/save2.ckpt')
    print('已经提取完毕：\nW=\n%s\nb=\n%s' % (sess.run(W), sess.run(b)))
