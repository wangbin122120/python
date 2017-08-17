import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(tf.add([[1], [2], [4]], [[4], [1], [3]]), axis=-1)))
    print(sess.run(tf.reduce_sum(tf.add([[1], [2], [4]], [[4], [1], [3]]), axis=0)))


