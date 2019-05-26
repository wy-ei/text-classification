## Text-CNN

- 论文：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

## 配置

```python
class_num=10    # 类别数 
embed_num=5000  # 需要等于字典大小
embed_dim=64    # 字向量维度
kernel_num=128  # 卷积核数量
kernel_size_list=[3,4,5] # 卷积核尺寸
dropout=0.5     # 置 0 的概率
```

## 基本原理

![image](https://user-images.githubusercontent.com/7794103/58327903-63a30180-7e63-11e9-9c82-acc55c8e0b21.png)

该模型的基本思想是对输入序列先做 Embedding，而后使用不同窗口大小的 1D Conv 提取特征，经过 MaxPooing1D 后 一个卷积核得到一个标量，最后全部拼接起来，得到一个向量，然后使用全连接层加 softmax 进行分类。