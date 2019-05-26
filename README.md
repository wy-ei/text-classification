## Text-Classification

使用 PyTorch 实现了以下几种文本分类模型：

#### Text-CNN

- 目录：[cnn](./cnn)
- 论文：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

#### Text-RCNN

- 目录：[rcnn](./rcnn)
- 论文: [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)

#### RNN-Attention

- 目录：[rnn-attention](./rnn-attention)
- 论文: [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174) - 简化版实现。

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

**1. 下载数据集**

下载数据集并解压至 `datasets` 目录下。

**2. 配置参数**

在 `mian.py` 中做适当调整，然后运行：

```
$ python main.py
```

## 运行结果：

这里并没有对文本进行过多的预处理，比如去除特殊符号，停用词等。另外直接采用了字作为特征，对于中文文本分类，感觉分词已经没有必要了。

以下都是用默认参数跑出来的结果，实验使用的 GPU 为 Tesla V100，如果要用 CPU 跑建议减少数据量，并限制文本长度。

### Text-CNN

```
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

2019-05-24 20:47:07,435 - test - acc: 0.9326
```

### Text-RCNN

```
2019-05-26 12:40:35,331 - epoch 1 - loss: 0.02 acc: 0.81 - val_loss: 0.00 val_acc: 0.90
2019-05-26 12:42:10,316 - epoch 2 - loss: 0.01 acc: 0.94 - val_loss: 0.01 val_acc: 0.90
2019-05-26 12:43:42,279 - epoch 3 - loss: 0.01 acc: 0.95 - val_loss: 0.00 val_acc: 0.93
2019-05-26 12:45:14,370 - epoch 4 - loss: 0.00 acc: 0.96 - val_loss: 0.00 val_acc: 0.91
2019-05-26 12:46:46,713 - epoch 5 - loss: 0.00 acc: 0.96 - val_loss: 0.00 val_acc: 0.94

2019-05-26 12:46:51,099 - test - acc: 0.95
```

相对 CNN 而言，RCNN 训练花费时间更多，RCNN 训练一个 epoch 可以让 CNN 训练 10 个 epoch。另外 RCNN 需要的 epoch 数相对较少，这里第一个 epoch 结束后，验证集上就达到了 90% 的准确度。

### RNN-Attention

```
2019-05-26 12:55:42,786 - epoch 1 - loss: 0.03 acc: 0.66 - val_loss: 0.01 val_acc: 0.80
2019-05-26 12:57:04,999 - epoch 2 - loss: 0.01 acc: 0.87 - val_loss: 0.01 val_acc: 0.84
2019-05-26 12:58:36,714 - epoch 3 - loss: 0.01 acc: 0.91 - val_loss: 0.01 val_acc: 0.88
2019-05-26 13:00:08,892 - epoch 4 - loss: 0.01 acc: 0.93 - val_loss: 0.01 val_acc: 0.89
2019-05-26 13:01:41,746 - epoch 5 - loss: 0.01 acc: 0.94 - val_loss: 0.00 val_acc: 0.92

2019-05-26 13:01:47,011 - test - acc: 0.9212
```

### FastText

另外，我使用 [FastText](https://fasttext.cc/) 对该数据集进行了分类，发现分类准确度能轻松达到 99% 以上。这也表明，对于长文本分类问题，词袋模型就足够了。深度模型，在此简单任务上并没有优势。

```
F1-Score : 0.999400  Precision : 0.999800  Recall : 0.999000   __label__0
F1-Score : 0.995690  Precision : 0.997991  Recall : 0.993400   __label__5
F1-Score : 0.996396  Precision : 0.997395  Recall : 0.995400   __label__1
F1-Score : 0.998701  Precision : 0.998003  Recall : 0.999400   __label__2
F1-Score : 0.999000  Precision : 0.999400  Recall : 0.998600   __label__3
F1-Score : 0.983119  Precision : 0.987884  Recall : 0.978400   __label__8
F1-Score : 0.997598  Precision : 0.998397  Recall : 0.996800   __label__9
F1-Score : 0.985344  Precision : 0.975873  Recall : 0.995000   __label__4
F1-Score : 0.996898  Precision : 0.997597  Recall : 0.996200   __label__6
F1-Score : 0.998700  Precision : 0.998800  Recall : 0.998600   __label__7
N       50000
P@1     0.995
R@1     0.995
```
