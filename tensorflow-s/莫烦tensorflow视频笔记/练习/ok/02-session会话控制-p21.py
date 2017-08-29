import tensorflow as tf

a=tf.constant([[3,3]])
b=tf.constant([[2],
               [2]])
c=tf.matmul(a,b)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(c))