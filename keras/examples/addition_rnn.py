# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

执行序列学习来执行加法
输入：“535 + 61”
输出：“596”
填充使用重复的前哨字符（空格）处理

输入可以可选地反转，显示为在以下的许多任务中提高性能：
“学习执行”
http://arxiv.org/abs/1410.4615
和“序列学习与神经网络序列”
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
从理论上讲，它会在源和目标之间引入较短的依赖关系。

两位数倒数：
+一层LSTM（128 HN），5k训练样本= 55个时期的99％列车/测试精度

三位倒数：
+一层LSTM（128 HN），50k训练样本= 99％列车/测试精度在100个时期

四位倒数：
+一层LSTM（128 HN），400k训练实例= 20个时期的99％列车/测试精度

五位倒数：
+一层LSTM（128 HN），550k训练样本= 99％列车/测试精度在30个纪元
'''

from __future__ import print_function

import numpy as np
# 如果是keras，前面加上这两句
import tensorflow as tf
from six.moves import range

from keras import Sequential
from keras import layers

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True #开始不会给tensorflow全部gpu资源 而是按需增加
sess = tf.Session(config=config)


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    给定一组字符：
     +将它们编码为一个热的整数表示
     +将一个热整数表达式解码为其字符输出
     +将一个概率向量解码为其字符输出
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
            初始化字符表。

         ＃参数
             字符：可以在输入中出现的字符。
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
           一个热编码给出字符串C.

         ＃参数
             num_rows：返回的一个热编码中的行数。 这是
                 用于保持每个数据的行数相同。
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset.
#模型和数据集的参数。
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
# 输入的最大长度为'int + int'（例如'345 + 678'）。 最大长度
# int is DIGITS。
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
# 所有的数字，加上填充的符号和空格。
chars = '0123456789+ '
ctable = CharacterTable(chars)
'''
ctable 通过转换后形成两个字典序列：
char to indices:
{' ': 0, '+': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11}
indices to char:
{0: ' ', 1: '+', 2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9'}
'''

questions = []
expected = []
seen = set()
print('Generating data...')
# 第1步：生成测试用例
# while循环将生成加法问题及其答案，分别保存在 questions 和 expected 中。毫秒级。
while len(questions) < TRAINING_SIZE:
    #从'0123456789' 随机抽取(1, DIGITS + 1)位数的字符组成数字
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()  # 随机结果，比如是(3, 1)
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    # 跳过我们已经看到的任何附加问题
    # 也可以跳过任何这样的x + Y == Y + x（因此排序）。
    key = tuple(sorted((a, b))) # (1, 3)
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    #用空格填充数据，使其始终为MAXLEN。
    q = '{}+{}'.format(a, b)    #'3+1'
    query = q + ' ' * (MAXLEN - len(q)) #'3+1    '
    ans = str(a + b)    #'4'
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans)) #'4   '
    if INVERT:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)

        query = query[::-1]
    #收集问题和答案
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))


# 第2步：创建RNN模型 并切分训练集、测试集
# 创建x,y，保存问题和答案，x:50000个问题，7位字符串，每一位对应12个字符
print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool) #shape (50000, 7, 12)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool) # shape (50000, 4, 12)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y)) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,...,49999])
np.random.shuffle(indices)  #打乱indices中元素顺序，这两步相当于定义一个随机数组。
x = x[indices]     #打乱x,y的元素顺序
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]  #切分训练集和测试集
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM  # 选择 LSTM 网络
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()  #初始化
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars)))) #设置RNN网络
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 1))  # 重复次数
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    # ＃通过将return_sequences设置为True，不仅返回最后一个输出
    #      ＃所有输出到目前为止（num_samples，timesteps，
    #      ＃output_dim）。 这是必要的，因为TimeDistributed在以下期望
    #      ＃第一个维度作为时间步长。
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# 第3步：训练与预测
# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
# ＃将密集层应用到输入的每个时间片。 对于每一步
# ＃输出序列，决定应选择哪个字符。
model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.#控制循环次数，迭代50次
for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    #训练，期间打印一堆堆的就是由这个计算过程产生
    # fit的说明在：D:\Anaconda3\Lib\site-packages\keras\models.py -->fit
    # 但真正fit实现的程序是：D:\Anaconda3\Lib\site-packages\keras\engine\training.py --> fit
    model.fit(x_train, y_train,  #数组训练数据以及目标数据的Numpy数组，
              batch_size=BATCH_SIZE, #梯度更新的样本数。
              epochs=1,     #迭代次数在训练数据阵列上。
              validation_data=(x_val, y_val)) #要评估的数据模型不对这些数据进行培训。
    # Select 10 samples from the validation set at random so we can visualize
    # errors. 验证准确性
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=" ")
        else:
            print(colors.fail + '☒' + colors.close, end=" ")
        print(guess)
        print('---')
