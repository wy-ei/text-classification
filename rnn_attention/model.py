import torch
import torch.nn as nn
import torch.nn.functional as F


class Bi_RNN_ATTN(nn.Module):
    def __init__(self,
                 device=None,
                 embed_size=None,
                 class_num=None,
                 embed_dim=64,
                 hidden_size= 256,
                 rnn_model='lstm',
                 dropout=0.5):
        super(Bi_RNN_ATTN, self).__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        self.rnn_model = rnn_model
        
        self.word_embedding = nn.Embedding(embed_size, embed_dim)

        if rnn_model == 'lstm':
            RNN = nn.LSTM
        else:
            RNN = nn.GRU
        
        self.rnn = RNN(input_size=embed_dim,
                            hidden_size=hidden_size // 2,
                            num_layers=1, bidirectional=True)
        
        self.w = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.output_fc = nn.Linear(hidden_size, class_num)

        self.rnn_dropout = nn.Dropout(dropout)
    
    def attention(self, outputs):
        """
        outputs 的 shape 为 (seq_len, batch_size, hidden_size)
        要计算 w 和 M 的 矩阵乘法，需要对 outputs 做转置，
        转置后，其 shape 为 (batch_size, seq_len, hidden_size)
        
        而 w 的 shape 为 (1, 1, hidden_size), 
        bmm(M, w) 的 shape 为 (batch_size, seq_len, 1)
        """  
        # outputs.shape -> (seq_len, batch_size, hidden_size)
        
        # shape: (batch_size, seq_len, hidden_size)
        outputs = torch.transpose(outputs, 0, 1)
        
        # shape: (batch_size, seq_len, hidden_size)
        M = torch.tanh(outputs)
        
        # shape: (batch_size, seq_len)
        alpha = F.softmax(torch.sum(M * self.w, dim=2), 1)
        
        # shape: (batch_size, 1, seq_len)
        alpha = alpha.unsqueeze(1)
        
        # shape: (batch_size, 1, hidden_size)
        r = torch.bmm(alpha, outputs)
        
        # shape: (batch_size, hidden_size)
        r = r.view(-1, self.hidden_size)
        
        return torch.tanh(r)
    
    def forward(self, sequences):
        batch_size = sequences.shape[0]
        
        # (batch_size, sequence_len, embedding_dim)
        embeds = self.word_embedding(sequences)

        # (sequence_len, batch_size, embedding_dim)
        embeds = torch.transpose(embeds,0,1)
        
        # ----------------LSTM---------------
        
        # 初始 hidden 的 shape 为 (batch_size, self.hidden_size / 2)
        h0 = torch.randn(2, batch_size, self.hidden_size // 2, device=self.device)
        if self.rnn_model == 'lstm':
            c0 = torch.randn(2, batch_size, self.hidden_size // 2, device=self.device)
            hidden = (h0, c0)
        else:
            hidden = h0
        # outputs 的 shape 为 (seq_len, batch, num_directions * hidden_size)
        # 这里采用的是双向 lstm，因此这里 output 的 shape 为 (seq_len, batch_size, self.hidden_size)
        
        # hidden 的 shape 为 (num_layers * num_directions, batch, hidden_size) 其中 num_layers = 1
        # num_directions = 2, hidden_size = self.hidden_size / 2, 此处并没有使用 hidden
        outputs, _ = self.rnn(embeds, hidden)
        
        # dropout 后 shape 不变
        outputs = self.rnn_dropout(outputs)
        
        # shape 为 (batch_size, hidden_size)
        h = self.attention(outputs)
        
        # shape 为 (batch_size, output_size)
        z = self.output_fc(h)     
        
        return z