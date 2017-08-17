
'''
tensorflow 提供的激励函数（83种）都在tf.nn下：http://devdocs.io/tensorflow~python/tf/nn
常用的是：
tf.nn -> relu,relu6,elu,softplus,softsign,dropout,bias_add
此外还有：
tf.   -> sigmoid,tanh
每种函数都可解决特定的问题，需要尽可能的了解每种函数的用法。

------------------------------------------83种激励函数-------------------------------------------------------------
all_candidate_sampler                       Generate the set of all classes.
                                            生成所有类的集合。
atrous_conv2d                               Atrous convolution (a.k.a. convolution with holes or dilated convolution).
                                            Atrous卷积（a.k.a.卷积孔或扩张卷积）。
atrous_conv2d_transpose                     The transpose of atrous_conv2d.
                                            atrous_conv2d的转置。
avg_pool                                    Performs the average pooling on the input.
                                            在输入上执行平均池。
avg_pool3d                                  Performs 3D average pooling on the input.
                                            在输入上执行3D平均池。
batch_norm_with_global_normalization        Batch normalization.
                                            批量归一化
batch_normalization                         Batch normalization.
                                            批量归一化
bias_add                                    Adds bias to value.
                                            增加价值的偏见。
bidirectional_dynamic_rnn                   Creates a dynamic version of bidirectional recurrent neural network.
                                            创建双向循环神经网络的动态版本。
compute_accidental_hits                     Compute the position ids in sampled_candidates matching true_classes.
                                            计算匹配true_classes的sampled_candidates中的位置id。
conv1d                                      Computes a 1-D convolution given 3-D input and filter tensors.
                                            计算给定3-D输入和滤波张量的1-D卷积。
conv2d                                      Computes a 2-D convolution given 4-D input and filter tensors.
                                            计算给定4-D输入和滤波张量的2-D卷积。
conv2d_backprop_filter                      Computes the gradients of convolution with respect to the filter.
                                            计算相对于滤波器的卷积梯度。
conv2d_backprop_input                       Computes the gradients of convolution with respect to the input.
                                            计算相对于输入的卷积梯度。
conv2d_transpose                            The transpose of conv2d.
                                            conv2d的转置。
conv3d                                      Computes a 3-D convolution given 5-D input and filter tensors.
                                            计算给定5-D输入和滤波张量的3-D卷积。
conv3d_backprop_filter_v2                   Computes the gradients of 3-D convolution with respect to the filter.
                                            计算相对于滤波器的3-D卷积的梯度。
conv3d_transpose                            The transpose of conv3d.
                                            conv3d的转置。
convolution                                 Computes sums of N-D convolutions (actually cross-correlation).
                                            计算N-D卷积（实际互相关）的和。
crelu                                       Computes Concatenated ReLU.
                                            计算并列ReLU。
ctc_beam_search_decoder                     Performs beam search decoding on the logits given in input.
                                            对输入中给出的逻辑进行波束搜索解码。
ctc_greedy_decoder                          Performs greedy decoding on the logits given in input (best path).
                                            对输入（最佳路径）中给出的逻辑执行贪婪解码。
ctc_loss                                    Computes the CTC (Connectionist Temporal Classification) Loss.
                                            计算CTC（连接时间分类）损失。
depthwise_conv2d                            Depthwise 2-D convolution.
                                            深度2-D卷积。
depthwise_conv2d_native                     Computes a 2-D depthwise convolution given 4-D input and filter tensors.
                                            计算给定4-D输入和滤波张量的2-D深度卷积。
depthwise_conv2d_native_backprop_filter     Computes the gradients of depthwise convolution with respect to the filter.
                                            计算相对于滤波器的深度卷积的梯度。
depthwise_conv2d_native_backprop_input      Computes the gradients of depthwise convolution with respect to the input.
                                            计算相对于输入的深度卷积的梯度。
dilation2d                                  Computes the grayscale dilation of 4-D input and 3-D filter tensors.
                                            计算4-D输入和3-D滤波器张量的灰度扩展。
dropout                                     Computes dropout.
                                            计算辍学。
dynamic_rnn                                 Creates a recurrent neural network specified by RNNCell cell.
                                            创建由RNNCell细胞指定的复发神经网络。
elu                                         Computes exponential linear: exp(features) - 1 if < 0, features otherwise.
                                            计算指数线性：exp（特征） - 如果<0，则具有其他特征。
embedding_lookup                            Looks up ids in a list of embedding tensors.
                                            在嵌入张量列表中查找ids。
embedding_lookup_sparse                     Computes embeddings for the given ids and weights.
                                            计算给定ids和权重的嵌入。
erosion2d                                   Computes the grayscale erosion of 4-D value and 3-D kernel tensors.
                                            计算4-D值和3-D内核张量的灰度侵蚀。
fixed_unigram_candidate_sampler             Samples a set of classes using the provided (fixed) base distribution.
                                            使用提供（固定）基本分布样本一组。
fractional_avg_pool                         Performs fractional average pooling on the input.
                                            在输入上执行小数平均池。
fractional_max_pool                         Performs fractional max pooling on the input.
                                            在输入上执行分数最大池。
fused_batch_norm                            Batch normalization.
                                            批量归一化
in_top_k                                    Says whether the targets are in the top K predictions.
                                            说明这些目标是否在K的前期预测中。
l2_loss                                     L2 Loss.
                                            L2损失。
l2_normalize                                Normalizes along dimension dim using an L2 norm.
                                            使用L2范数沿尺寸dim进行归一化。
learned_unigram_candidate_sampler           Samples a set of classes from a distribution learned during training.
                                            从培训中学习的分布中抽取一组课程。
local_response_normalization                Local Response Normalization.
                                            本地响应规范化。
log_poisson_loss                            Computes log Poisson loss given log_input.
                                            计算log_input的日志泊松损失。
log_softmax                                 Computes log softmax activations.
                                            计算log softmax激活。
log_uniform_candidate_sampler               Samples a set of classes using a log-uniform (Zipfian) base distribution.
                                            使用对数统一（Zipfian）基本分布对一组类进行抽样。
lrn                                         Local Response Normalization.
                                            本地响应规范化。
max_pool                                    Performs the max pooling on the input.
                                            在输入上执行最大池数。
max_pool3d                                  Performs 3D max pooling on the input.
                                            在输入上执行3D max pooling。
max_pool_with_argmax                        Performs max pooling on the input and outputs both max values and indices.
                                            在输入上执行最大汇总，并输出最大值和索引。
moments                                     Calculate the mean and variance of x.
                                            计算x的平均值和方差。
nce_loss                                    Computes and returns the noise-contrastive estimation training loss.
                                            计算并返回噪声对比估计训练损失。
normalize_moments                           Calculate the mean and variance of based on the sufficient statistics.
                                            根据足够的统计量计算平均值和方差。
pool                                        Performs an N-D pooling operation.
                                            执行N-D池操作。
quantized_avg_pool                          Produces the average pool of the input tensor for quantized types.
                                            生成量化类型的输入张量的平均池。
quantized_conv2d                            Computes a 2D convolution given quantized 4D input and filter tensors.
                                            计算给定量化4D输入和滤波张量的2D卷积。
quantized_max_pool                          Produces the max pool of the input tensor for quantized types.
                                            为量化类型生成输入张量的最大池。
quantized_relu_x                            Computes Quantized Rectified Linear X: min(max(features, 0), max_value)
                                            计算量化整流线性X：min（max（features，0），max_value）
raw_rnn                                     Creates an RNN specified by RNNCell cell and loop function loop_fn.
                                            创建由RNNCell单元格指定的RNN和循环函数loop_fn。
relu                                        Computes rectified linear: max(features, 0).
                                            计算整流线性：max（features，0）。
relu6                                       Computes Rectified Linear 6: min(max(features, 0), 6).
                                            计算整流线性6：最小（最大（特征，0），6）。
relu_layer                                  Computes Relu(x * weight + biases).
                                            计算Relu（x *权重+偏差）。
sampled_softmax_loss                        Computes and returns the sampled softmax training loss.
                                            计算并返回采样的softmax
separable_conv2d                            2-D convolution with separable filters.
                                            2-D卷积与可分离的过滤器。
sigmoid                                     Computes sigmoid of x element-wise.
                                            计算x元素的sigmoid。
sigmoid_cross_entropy_with_logits           Computes sigmoid cross entropy given logits.
                                            计算给定逻辑的S形交叉熵。
softmax                                     Computes softmax activations.
                                            计算softmax激活。
softmax_cross_entropy_with_logits           Computes softmax cross entropy between logits and labels.
                                            计算对数和标签之间的softmax交叉熵。
softplus                                    Computes softplus: log(exp(features) + 1).
                                            计算softplus：log（exp（features）+ 1）。
softsign                                    Computes softsign: features / (abs(features) + 1).
                                            计算softsign：features /（abs（功能）+ 1）。
sparse_softmax_cross_entropy_with_logits    Computes sparse softmax cross entropy between logits and labels.
                                            计算对数和标签之间的稀疏softmax交叉熵。
static_bidirectional_rnn                    Creates a bidirectional recurrent neural network.
                                            创建双向循环神经网络。
static_rnn                                  Creates a recurrent neural network specified by RNNCell cell.
                                            创建由RNNCell细胞指定的复发神经网络。
static_state_saving_rnn                     RNN that accepts a state saver for time-truncated RNN calculation.
                                            RNN接受状态保护程序进行时间截断的RNN计算。
sufficient_statistics                       Calculate the sufficient statistics for the mean and variance of x.
                                            计算x的均值和方差的足够统计。
tanh                                        Computes hyperbolic tangent of x element-wise.
                                            计算x元素的双曲正切。
top_k                                       Finds values and indices of the k largest entries for the last dimension.
                                            查找最后一个维度的k个最大条目的值和索引。
uniform_candidate_sampler                   Samples a set of classes using a uniform base distribution.
                                            使用均匀的基本分布样本一组。
weighted_cross_entropy_with_logits          Computes a weighted cross entropy.
                                            计算加权交叉熵。
weighted_moments                            Returns the frequency-weighted mean and variance of x.
                                            返回x的频率加权均值和方差。
with_space_to_batch                         Performs op on the space-to-batch representation of input.
                                            执行输入的空对多表示。
xw_plus_b                                   Computes matmul(x, weights) + biases.
                                            计算matmul（x，权重）+偏差。
zero_fraction                               Returns the fraction of zeros in value.
                                            返回值的零分数。

'''