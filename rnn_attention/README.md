## Bi-RNN-Attention

- 论文: [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174)

## 配置

```python
class_num=10     # 类别数 
embed_num=5000   # 需要等于字典大小
embed_dim=64     # 字向量维度
device=...       # 训练使用的 device，如 `torch.device('cuda')`
dropout=0.5      # RNN 输出置 0 的概率
rnn_model='lstm' # RNN 使用的模型，默认为 LSTM 也可以是 GRU
```

## 基本原理

![image](https://user-images.githubusercontent.com/7794103/58372118-bd7ef680-7f4b-11e9-806d-03ae6ab9559c.png)

在原论文中，是对整篇文章做编码，先对单词做 Attention 完成对句子的编码，在对句子做 Attention 完成对整个文档的编码。

![image](https://user-images.githubusercontent.com/7794103/58372145-21a1ba80-7f4c-11e9-8e80-ac5974734550.png)

这里，对模型进行了简化，直接对单词做 Attention 完成对整个文档的编码。


RNN 的每个时间步会得到一个隐状态，整个序列处理完成后会得到隐状态列表 `H = [h_0, h_1, h_2, ..., h_n]`，这里模型引入一个可学习的向量 w，用 w 和 h_i 计算 attention 的权重。Attention 的目的是提取最为重要的信息，这里对 RNN 的隐状态做 attention，其结果就是只会关注到个别几个隐状态，可能就是那些对分类最有帮助的隐状态。