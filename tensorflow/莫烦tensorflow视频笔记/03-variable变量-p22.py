import tensorflow as tf

state = tf.Variable(0, name='counter')  # 命名的好处等下看看
# print(state.name)  #counter:0
one = tf.constant(1)  # 常量

new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 将new_value当前的状态复制到state,使得state=new_value

# 若定义了Variable,那么必定有一步定义：
init = tf.global_variables_initializer()

