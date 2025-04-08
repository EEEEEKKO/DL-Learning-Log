import math
from typing import Optional
import torch
import torch.nn as nn
from Classic_Transformers.MultiHeadAttention import MultiHeadAttention


def get_slopes(n_heads: int):
    #get the closest power of 2 for the number of heads
    n = 2 ** math.floor(math.log2(n_heads))

    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1+n))

    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2*(n_heads-n), 2))
        m = torch.cat(m, m_hat)
    
    return m 

@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    '''
    mask:[seq_len_q, seq_len_k]
    return: [seq_len_q, seq_len_k, n_heads]
    '''
    m = get_slopes(n_heads).to(mask.device)
    distance = mask.cumsum(dim=-1)

    return distance[:,:,None] * m[None, None, :]

class ALiBiMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout: float):
        super().__init__(heads = heads, d_model = d_model, dropout = dropout)

        self.alibi_biases = None
