import torch
import torch.nn as nn
from typing import Optional, List

class Attention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads*d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        
        head_shape = x.shape[:-1]
        x = self.linear(x)

        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.5, bias: bool = True):
        super().__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads

        self.W_q = Attention(d_model, heads, self.d_k, bias)
        self.W_k = Attention(d_model, heads, self.d_k, bias)
        self.W_v = Attention(d_model, heads, self.d_k, bias)

        self.Dropout = nn.Dropout(dropout)
        self.W_o = nn.Linear(d_model, d_model)
    
    def get_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)

        return mask
    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        query = query.permute(1,2,0,3)
        key = key.permute(1,2,0,3)
        scores = torch.matmul(query, key.transpose(-2,-1)) / (self.d_k ** 0.5)
        scores = scores.permute(2,3,0,1)
        return scores
    
    def forward(self,  
                query: torch.Tensor, 
                key: torch.Tensor,
                value: torch.Tensor,
                mask : Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.get_mask(mask, query.shape, key.shape)
        
        #[seq_len, batch_size, heads, d_k]
        query = self.W_q(query) 
        key = self.W_k(key)
        value = self.W_v(value)
        
        #[seq_len_q, seq_len_k, batch_size, heads]
        scores = self.get_scores(query, key)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.softmax(scores)
        attn = self.Dropout(attn)


        # x = torch.matmul(attn.permute(2,3,0,1), value.permute(1,2,0,3)).permute(2,0,1,3)
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)

        x = x.reshape(seq_len, batch_size,-1)
        return self.W_o(x)



