## Text-CNN

使用 PyTorch 实现 [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) 中提出的文本分类方法。

## 数据集

此处使用的数据集来自 [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn) 作者整理的数据集。下载链接：https://pan.baidu.com/s/1hugrfRu 密码: qfud

该数据集共包含 10 个类别，每个类别有 6500 条数据。类别如下：

```
'体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐'
```

数据集划分如下：

- 训练集: 5000 * 10
- 验证集: 500 * 10
- 测试集: 1000 * 10

## 运行方法

下载数据集，并解压至 `datasets` 目录下，在 `main.py` 中做适当调整，然后运行：

```
$ python main.py
```

运行结果：

```
2019-05-24 20:45:03,204 - using device: cuda:7
2019-05-24 20:45:03,205 - load and preprocess data...
2019-05-24 20:45:15,800 - training...
2019-05-24 20:45:30,872 - epoch: 1 - loss: 0.06 acc: 0.65 - val_loss: 0.03 val_acc: 0.75
2019-05-24 20:45:41,568 - epoch: 2 - loss: 0.05 acc: 0.80 - val_loss: 0.03 val_acc: 0.77
2019-05-24 20:45:52,137 - epoch: 3 - loss: 0.05 acc: 0.82 - val_loss: 0.03 val_acc: 0.82
2019-05-24 20:46:02,975 - epoch: 4 - loss: 0.05 acc: 0.83 - val_loss: 0.03 val_acc: 0.78
2019-05-24 20:46:13,769 - epoch: 5 - loss: 0.05 acc: 0.83 - val_loss: 0.03 val_acc: 0.82
2019-05-24 20:46:24,514 - epoch: 6 - loss: 0.05 acc: 0.87 - val_loss: 0.02 val_acc: 0.90
2019-05-24 20:46:35,237 - epoch: 7 - loss: 0.05 acc: 0.92 - val_loss: 0.02 val_acc: 0.90
2019-05-24 20:46:45,801 - epoch: 8 - loss: 0.05 acc: 0.93 - val_loss: 0.02 val_acc: 0.91
2019-05-24 20:46:56,050 - epoch: 9 - loss: 0.05 acc: 0.93 - val_loss: 0.02 val_acc: 0.93
2019-05-24 20:47:06,771 - epoch: 10 - loss: 0.05 acc: 0.94 - val_loss: 0.02 val_acc: 0.94
2019-05-24 20:47:07,000 - predicting...
2019-05-24 20:47:07,435 - test - acc: 0.9326
```

这里并没有对文本进行过多的预处理，比如去除特殊符号，停用词等。另外直接采用了字作为特征，对于中文文本分类，感觉分词已经没有必要了。

另外我还使用了 LogisticRegression 作为 baseline 对此数据集进行了分类。实验表明在验证集上得到了 90.8% 的准确度。尽管使用了较大的正则化系数，模型仍然有些过拟合，在训练集上准确率达到 96.2%。

## 配置

```python
class_num=10    # 类别数 
embed_num=5000  # 字典大小
embed_dim=64    # 字向量维度
kernel_num=128  # 卷积核数量
kernel_size_list=[3,4,5] # 卷积核尺寸
dropout=0.5     # 置 0 的概率
```

## Text CNN 模型

![image](https://user-images.githubusercontent.com/7794103/58327903-63a30180-7e63-11e9-9c82-acc55c8e0b21.png)

该模型的基本思想是对输入序列先做 Embedding，而后使用不同窗口大小的 1D Conv 提取特征，经过 MaxPooing1D 后 一个卷积核得到一个标量，最后全部拼接起来，得到一个向量，然后使用全连接层加 softmax 进行分类。