import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    def __init__(self, num_blocks, hidden_size, dropout_rate):
        super(QuestionEncoder, self).__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.transformer_blocks = nn.ModuleList([nn.TransformerEncoderLayer(hidden_size,
                                                                            nn.MultiheadAttention(hidden_size),
                                                                            nn.Linear(hidden_size, hidden_size),
                                                                            dropout_rate) for _ in range(num_blocks)])
        self.transformer = nn.TransformerEncoder(self.transformer_blocks)
        self.embedding = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.embedding(x)
        return x
