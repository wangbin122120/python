import tensorflow as tf


'''tf.train下面有很多方法：http://devdocs.io/tensorflow~python/tf/train
常用的有如：
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.MomentumOptimizer      会考虑前一步的优化情况
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.RMSPropOptimizer

tf.AggregationMethod
tf.gradients()
tf.stop_gradient

------------------------具体的有56种：---------------------------
class AdadeltaOptimizer	                  Optimizer that implements the Adadelta algorithm.
                                          优化器实现Adadelta算法。
class AdagradDAOptimizer                  Adagrad Dual Averaging algorithm for sparse linear models.
                                          用于稀疏线性模型的Adagrad双平均算法。
class AdagradOptimizer                    Optimizer that implements the Adagrad algorithm.
                                          优化器实现Adagrad算法。
class AdamOptimizer                       Optimizer that implements the Adam algorithm.
                                          实现Adam算法的优化器。
class BytesList
class CheckpointSaverHook                 Saves checkpoints every N steps or seconds.
                                          每N个步骤或秒保存检查点。
class CheckpointSaverListener             Interface for listeners that take action before or after checkpoint save.
                                          在检查点保存之前或之后采取行动的侦听器的接口。
class ChiefSessionCreator                 Creates a tf.Session for a chief.
                                          为酋长创建一个tf.Session。
class ClusterDef
class ClusterSpec                         Represents a cluster as a set of "tasks", organized into "jobs".
                                          将集群表示为一组“任务”，组织成“工作”。
class Coordinator                         A coordinator for threads.
                                          线程协调员。
class Example
class ExponentialMovingAverage            Maintains moving averages of variables by employing an exponential decay.
                                          通过使用指数衰减来维持变量的移动平均值。
class Feature
class FeatureList
class FeatureLists
class Features
class FeedFnHook                          Runs feed_fn and sets the feed_dict accordingly.
                                          运行feed_fn并相应地设置feed_dict。
class FinalOpsHook                        A hook which evaluates Tensors at the end of a session.
                                          在会话结束时评估Tensors的钩子。
class FloatList
class FtrlOptimizer                       Optimizer that implements the FTRL algorithm.
                                          实现FTRL算法的优化器。
class GlobalStepWaiterHook                Delays execution until global step reaches wait_until_step.
                                          延迟执行，直到全局步骤达到wait_until_step。
class GradientDescentOptimizer            Optimizer that implements the gradient descent algorithm.
                                          实现梯度下降算法的优化器。
class Int64List
class JobDef
class LoggingTensorHook                   Prints the given tensors once every N local steps or once every N seconds.
                                          每N个本地步骤或每N秒一次打印给定的张量。
class LooperThread                        A thread that runs code repeatedly, optionally on a timer.
                                          重复运行代码的线程，可选地在定时器上运行。
class MomentumOptimizer                   Optimizer that implements the Momentum algorithm.
                                          Optimizer实现了Momentum算法。
class MonitoredSession                    Session-like object that handles initialization, recovery and hooks.
                                          处理初始化，恢复和挂钩的类Session对象。
class NanLossDuringTrainingError
class NanTensorHook                       Monitors the loss tensor and stops training if loss is NaN.
                                          监视损失张量并停止训练，如果损失是NaN。
class Optimizer                           Base class for optimizers.
                                          优化器的基类。
class ProximalAdagradOptimizer            Optimizer that implements the Proximal Adagrad algorithm.
                                          优化器实现了近似Adagrad算法。
class ProximalGradientDescentOptimizer    Optimizer that implements the proximal gradient descent algorithm.
                                          实现近端梯度下降算法的优化器。
class QueueRunner                         Holds a list of enqueue operations for a queue, each to be run in a thread.
                                          保存一个队列的入队操作列表，每个队列在线程中运行。
class RMSPropOptimizer                    Optimizer that implements the RMSProp algorithm.
                                          Optimizer实现RMSProp算法。
class Saver                               Saves and restores variables.
                                          保存和恢复变量。
class SaverDef                            ????
class Scaffold                            Structure to create or gather pieces commonly needed to train a model.
                                          用于创建或收集通常需要训练模型的构件的结构。
class SecondOrStepTimer                   Timer that triggers at most once every N seconds or once every N steps.
                                          定时器每N秒触发一次或每N个步骤触发一次。
class SequenceExample                     ????
class Server                              An in-process TensorFlow server, for use in distributed training.
                                          一个在进程中的TensorFlow服务器，用于分布式培训。
class ServerDef                           ????
class SessionCreator                      A factory for tf.Session.
                                          一个工厂为tf.Session。
class SessionManager                      Training helper that restores from checkpoint and creates session.
                                          培训助手从检查点恢复并创建会话。
class SessionRunArgs                      Represents arguments to be added to a Session.run() call.
                                          表示要添加到Session.run（）调用中的参数。
class SessionRunContext                   Provides information about the session.run() call being made.
                                          提供有关session.run（）调用的信息。
class SessionRunHook                      Hook to extend calls to MonitoredSession.run().
                                          挂钩将呼叫扩展到MonitoredSession.run（）。
class SessionRunValues                    Contains the results of Session.run().
                                          包含Session.run（）的结果。
class SingularMonitoredSession            Session-like object that handles initialization, restoring, and hooks.
                                          处理初始化，恢复和挂接的类Session对象。
class StepCounterHook                     Hook that counts steps per second.
                                          钩子每秒钟计数。
class StopAtStepHook                      Hook that requests stop at a specified step.
                                          挂钩请求在指定步骤停止。
class SummarySaverHook                    Saves summaries every N steps.
                                          每N个步骤保存摘要。
class Supervisor                          A training helper that checkpoints models and computes summaries.
                                          一个训练助手，检查点建模和计算摘要。
class SyncReplicasOptimizer               Class to synchronize, aggregate gradients and pass them to the optimizer.
                                          要同步的类，聚合渐变并将其传递给优化器。
class WorkerSessionCreator                Creates a tf.Session for a worker.
                                          为工作者创建一个tf.Session。

'''