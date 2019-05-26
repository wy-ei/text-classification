## RCNN

- 论文: [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)

## 配置

```python
class_num=10     # 类别数 
embed_num=5000   # 需要等于字典大小
embed_dim=64     # 字向量维度
kernel_num=128   # 卷积核数量
device=...       # 训练使用的 device，如 `torch.device('cuda')`
rnn_model='lstm' # RNN 使用的模型，默认为 LSTM 也可以是 GRU
dropout=0.5      # RNN 输出置 0 的概率
```

## 基本原理

![image](https://user-images.githubusercontent.com/7794103/58369441-da580180-7f2c-11e9-9677-5646ee49e406.png)


用 RNN 做分类的传统方法是，将整个句子或文档用 RNN 编码，使用 RNN 最后一个时间步的输出送入全连接层来进行分类。在处理长序列时，这样 RNN 往往会更偏向序列靠后的词，而且只能得到一个方向的编码结果。

双向的 RNN 能够改善此问题，但人们发现 RNN 中间状态也应该得到充分的应用。这里作者使用双向的 RNN 对每个词进行编码，然后将编码结果和该词的词向量拼接，最终整个序列进行 max pooling。然后使用全连接层进行分类。

词向量的每一个维度都表征着一个词的某种特征。RNN 的每一步会得出的隐状态，这个隐状态的每一维也能代表词的某个特征。RCNN 的想法，以我的理解，就是充分利用这些特征，而这些特征经过 max pooling 后，就保留了特征最强的值，max pooling 后得到的向量就充分表征了输入序列的特征。