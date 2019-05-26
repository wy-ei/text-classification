import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self,
                 class_num=None,
                 embed_size=None,
                 embed_dim=64,
                 kernel_num=128,
                 kernel_size_list=(3,4,5),
                 dropout=0.5):
        
        super(TextCNN, self).__init__()
                
        self.embedding = nn.Embedding(embed_size, embed_dim)
        
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(embed_dim, kernel_num, kernel_size)
                for kernel_size in kernel_size_list
        ])
        
        self.linear = nn.Linear(kernel_num * len(kernel_size_list), class_num)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x.shape is (batch, word_nums)
        
        # after embedding x.shape is (batch, word_nums, embed_dim)
        x = self.embedding(x)
        
        # since the input of conv1d require shape: (batch, in_channels, in_length)
        # here in_channels is embed_dim, in_length is word_nums
        # we should tranpose x into shape: (batch, embed_dim, word_nums)
        x = x.transpose(1, 2)
        
        # after conv1d the shape become: (batch, kernel_num, out_length)
        # here out_length = word_nums - kernel_size + 1
        x = [F.relu(conv1d(x)) for conv1d in self.conv1d_list]

        # pooling apply on 3th dimension, window size is the length of 3th dim
        # after pooling the convert to (batch, kernel_num, 1)
        # squeeze is requred to remove the 3th dimention
        x = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]

        # shape: (batch, kernel_num * len(kernel_size_list))
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        
        # shape: (batch, class_num)
        x = self.linear(x)
        
        return F.softmax(x, dim=1)