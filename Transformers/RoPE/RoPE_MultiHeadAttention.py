import torch
import torch.nn as nn
import math
from Classic_Transformers.MultiHeadAttention import MultiHeadAttention

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.d = d
        self.base = base
        self.cos_cached = None
        self.sin_cached = None
    
    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        seq_len = x.shape[0]

        theta = 1. / (self.base ** (torch.arange(0,self.d,2).float()/self.d)).to(x.device)

        seq_index = torch.arange(seq_len).float().to(x.device)

        idx_theta = torch.einsum('n,d->nd', seq_index, theta)

        idx_theta = torch.cat((idx_theta, idx_theta), dim=1)

        self.cos_cached = idx_theta.cos()[:, None, None, :]
        self.sin_cached = idx_theta.sin()[:, None, None, :]
    
    def _neg_half(self, x:torch.Tensor):
        
        d_2 = self.d // 2
        return torch.cat((-x[:, :, :, :d_2], x[:, :, :, d_2:]), dim=-1)
    
    def forward(self, x: torch.Tensor):
        """
        x : [seq_len, batch_size, n_heads, d_model]
        """
        self._build_cache(x)

        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        return torch.cat((x_rope, x_pass), dim=-1)

class RoPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.5, bias: bool = True, rope_percentage: float = 0.5):
        super().__init__(d_model, heads, dropout, bias)

        d_rope = int(d_model * rope_percentage)
        self.query_rope = RotaryPositionalEmbeddings(d_rope)
        self.key_rope = RotaryPositionalEmbeddings(d_rope)
    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        
        return torch.einsum('ibhd, jbhd -> ijbh', self.query_rope(query), self.key_rope(key)) / math.sqrt(self.d_k)


def test_rope_multihead_attention():
    d_model = 256
    heads = 16
    dropout = 0.5
    bias = True
    rope_percentage = 0.5
    batch_size = 2
    seq_len = 10

    query = torch.randn(batch_size, seq_len, heads, d_model)
    key = torch.randn(batch_size, seq_len, heads, d_model)

    rope_multihead_attention = RoPEMultiHeadAttention(d_model, heads, dropout, bias, rope_percentage)
    
    scores = rope_multihead_attention.get_scores(query, key)

    print(scores.shape)
if __name__ == "__main__":
    test_rope_multihead_attention()

