import torch
import torch.nn as nn
import torch.nn.functional as F
from model.counting import Counter
from torch.nn.utils.weight_norm import weight_norm
from block import fusions

import torch
import torch.nn as nn


class MCB(nn.Module):
    def __init__(self, input_size, num_heads, dropout=0.1):
        super(MCB, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.fc_query = nn.Linear(input_size, input_size, bias=False)
        self.fc_key = nn.Linear(input_size, input_size, bias=False)
        self.fc_value = nn.Linear(input_size, input_size, bias=False)
        self.fc_out = nn.Linear(input_size, input_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.fc_query(query).view(batch_size, -1, self.num_heads, self.input_size // self.num_heads).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        key = self.fc_key(key).view(batch_size, -1, self.num_heads, self.input_size // self.num_heads).permute(0, 2, 3,
                                                                                                               1)
        value = self.fc_value(value).view(batch_size, -1, self.num_heads, self.input_size // self.num_heads).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)

        scores = torch.matmul(query, key) / (self.input_size // self.num_heads) ** 0.5
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = torch.matmul(scores, value)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_size)
        out = self.fc_out(out)

        return out
