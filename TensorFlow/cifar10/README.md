CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/


文件	作用

关键硬件：
    100% 1620v4 3.6Ghz/4c
     50% 1080Ti 1.9Ghz/3584c/11GB
         ssd 850 pro
         ddr4 2133Mhz


1.模型训练
cifar10_train.py	在CPU或GPU上训练CIFAR-10的模型。
cifar10_multi_gpu_train.py	在多GPU上训练CIFAR-10的模型。

    cifar10.py	        建立CIFAR-10的模型。
        cifar10_input.py	读取本地CIFAR-10的二进制文件格式的内容。

训练速度：
2017-08-05 21:55:10.522042: step 124610, loss = 0.66 (8494.5 examples/sec; 0.015 sec/batch)


周期性的保存模型中的所有参数(大概十分钟一次)：
    D:\tmp\cifar10_train

        model.ckpt-97389.data-00000-of-00002
        model.ckpt-97389.data-00001-of-00002
        model.ckpt-97389.index
        model.ckpt-97389.meta

2.模型预测
cifar10_eval.py	评估CIFAR-10模型的预测性能。


