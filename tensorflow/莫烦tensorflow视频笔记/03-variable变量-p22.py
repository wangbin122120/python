import tensorflow as tf

state = tf.Variable(0, name='counter')  # 命名的好处等下看看
# print(state.name)  #counter:0
one = tf.constant(1)  # 常量

new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 将new_value当前的状态复制到state,使得state=new_value

# 若定义了Variable,那么必定有两步：第一步是定义（如下），第二步是激活sess.run(init)。
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) #必须要run()之后才会出结果

    '''输出结果：
    1
    2
    3
    '''

#如果需要初始化为0，可以这样：
    print(sess.run(tf.assign(state, tf.constant(0)))) #0 ,state 就此变成0。
    for _ in range(3):
        print(sess.run(update))

    '''输出结果：
    0
    1
    2
    3
    '''


    state = tf.Variable(0, name='counter')      #即使重新设置0，但这样得到的也是另外一个变量
    print(state.name)       #counter_1:0  不同于开始的 counter:0
    for _ in range(3):
        print(sess.run(update))

    '''输出结果：
    counter_1:0
    4
    5
    6
    
    Process finished with exit code 0
    '''