# HWGD
## 激活值的quantize
假设激活值基本服从正太分布，解决了需要通过迭代来选取优化的参数的问题。使用非均匀的quantize。

## forward
前向传播就是一个离散化的relu

##backward
因为离散化的relu不可能到正无穷，所以反向传播大于0的部分梯度为1会mismatch。因此反向传播使用拼接的函数。

（未完待补充