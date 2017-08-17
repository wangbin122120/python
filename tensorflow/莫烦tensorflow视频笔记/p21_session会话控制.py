import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1, matrix2)  # matmul = matrix multiply

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)  # [[12]]
sess.close()

# method 2
with tf.Session() as sess:  # open session ,not need close session .
    result = sess.run(product)
    print(result)  # [[12]]
