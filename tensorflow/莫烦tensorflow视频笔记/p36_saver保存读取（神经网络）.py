
import tensorflow as tf
# 限制显卡显存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True #开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

'''
在训练过程中，经常防止意外中断，使用Saver()保存训练过程中的参数，以便下次继续。
'''
# ------------------------- 保存过程 -----------------------------
#
# #定义几个tensor
# W=tf.Variable([[1,2,3],[2,3,4]],dtype=tf.float32)  #在保存和读取的时候都要定义好dtype,以免出现数据类型冲突
# b=tf.Variable([[1,2,3]],dtype=tf.float32)
#
# #定义好Saver
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     save_path = saver.save(sess,'/tmp/tensorflow/mofan_logs/p36/save.ckpt') # 文件后缀.ckpt是官方约定的，
#     #提取的时候，仅仅将save()替换成restore()
#     #保存完毕后会生成4个文件
#     '''
#     checkpoint
#     save.ckpt.data-00000-of-00001
#     save.ckpt.index
#     save.ckpt.meta
#     '''
#     print('已经保存完毕：',save_path)
#

# # ------------------------- 提取过程 -----------------------------
# #注意目前Saver()只能保存和导出Variable()变量，所以对于整个神经网络框架程序，还是需要重新设置，就是可以保存数据。
# #定义几个tensor
# import numpy as np #用于初始化比较方便。
# W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32)  #在保存和读取的时候都要定义好dtype,以免出现数据类型冲突
# b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32)
#
# #定义好Saver,还是用saver提取
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     # 在提取的时候，不需要初始化，因为restore自然会做.
#     #sess.run(tf.global_variables_initializer())
#
#     saver.restore(sess,'/tmp/tensorflow/mofan_logs/p36/save.ckpt') # 文件后缀.ckpt是官方约定的，
#     #打印之前保存的结果。
#     print('已经提取完毕：\nW=\n%s\nb=\n%s'%(sess.run(W),sess.run(b)))
#     '''运行结果
# 已经提取完毕：
# W=
# [[ 1.  2.  3.]
#  [ 2.  3.  4.]]
# b=
# [[ 1.  2.  3.]]
#     '''