import tensorflow as tf

input1 =tf.placeholder(tf.float32) # 需要传入数据，一般是float32，可以定义数组结构
input2 =tf.placeholder(tf.float32) # 需要传入数据，一般是float32

output =tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[4],input2:[5]}))  #有placeholder，所以需要传入值
    # 运行结果：[ 20.]