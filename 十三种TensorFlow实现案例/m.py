import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(tf.add([1, 2, 3], [4, 5, 6])))
    print(sess.run(tf.negative([1, 2, -4])))
    print(sess.run(tf.reduce_sum(tf.add([[1], [2], [4]], [[4], [1], [3]]), reduction_indices=0)))
    print(sess.run(tf.reduce_sum(tf.add([[1], [2], [4]], [[4], [1], [3]]), axis=0)))


