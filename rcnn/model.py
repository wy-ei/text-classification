import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):
    def __init__(self,
                 device=None,
                 class_num=None,
                 embed_size=None,
                 embed_dim=128,
                 hidden_size= 256,
                 rnn_model='lstm',
                 dropout=0.5):
        super(TextRCNN, self).__init__()
        
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
        
        self.output_fc = nn.Linear(hidden_size + embed_dim, class_num)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequences):
        batch_size = sequences.shape[0]
        
        # sequences.shape: (batch, sequence_len)

        # shape: (sequence_len, batch)
        sequences = sequences.transpose(0, 1)

        # (sequence_len, batch_size, embedding_dim)
        embeds = self.word_embedding(sequences)
        
        # ----------------RNN---------------
        
        # 初始 hidden 的 shape 为 (batch_size, self.hidden_size / 2)
        h0 = torch.randn(2, batch_size, self.hidden_size // 2, device=self.device)
        if self.rnn_model == 'lstm':
            c0 = torch.randn(2, batch_size, self.hidden_size // 2, device=self.device)
            hidden = (h0, c0)
        else:
            hidden = h0
        
        # outputs 的 shape 为 (sequence_len, batch, num_directions * hidden_size)
        outputs, _ = self.rnn(embeds, hidden)

        # shape: (sequence_len, batch, num_directions * hidden_size + embedding_size)
        x = torch.cat((outputs, embeds), dim=2)
        
        # (batch, num_directions * hidden_size + embedding_size, sequence_len)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)

        # (batch, num_directions * hidden_size + embedding_size)
        x = F.max_pool1d(x, x.shape[1]).squeeze(2)

        x = self.dropout(x)
        
        # shape 为 (batch_size, output_size)
        z = self.output_fc(x) 
        
        return z