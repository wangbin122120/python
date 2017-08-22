import tensorflow as tf
# 限制显卡显存
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True #开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)

#保存数据
W=tf.Variable([[1,2,3],[2,3,4]],dtype=tf.float32)
b=tf.Variable([1,2,3],dtype=tf.float32)
