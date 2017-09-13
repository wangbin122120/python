import tensorflow as tf

state = tf.Variable(0)
one = tf.constant(1)
add = tf.add(state, one)
update = tf.assign(state, add)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(add))
    print(sess.run(add))
    print(sess.run(add))
    print(sess.run(update))
    print(sess.run(update))
    print(sess.run(update))

    #如果需要初始化为0，可以这样：
    print(sess.run(tf.assign(state, tf.constant(0))))
    for _ in range(3):
        print(sess.run(update))
