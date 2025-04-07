import math
import torch
import torch.nn as nn
import numpy as np


def get_positional_encoding(max_len:int, d_model:int):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)

    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.5, max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer('PE', get_positional_encoding(max_len, d_model))
    
    def forward(self, x: torch.Tensor):
        pe = self.PE[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x

