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
        m_hat_0 = 2.0 ** (-4.0 / n) #decrease faster for the extra heads
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2*(n_heads-n), 2))
        m = torch.cat([m, m_hat])
    
    return m 

@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    '''
    mask:[seq_len_q, seq_len_k]
    return: [seq_len_q, seq_len_k, n_heads]
    '''
    m = get_slopes(n_heads).to(mask.device)
    distance = mask.cumsum(dim=-1)# calculate the relative position for each pair

    return distance[:,:,None] * m[None, None, :]

def get_alibi_biases_from_mask(n_heads: int, mask: torch.Tensor):
    '''
    mask:[seq_len_q, seq_len_k]
    return: [seq_len_q, seq_len_k, n_heads]
    '''
    m = get_slopes(n_heads).to(mask.device)
    distance = mask.cumsum(dim=-1)


class ALiBiMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout: float):
        super().__init__(heads = heads, d_model = d_model, dropout = dropout)

        #ALiBi cache
        self.alibi_biases = None
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        
        '''
        ALiBi only works with causal masks
        '''
        assert mask is not None
        assert mask.shape[0] == mask.shape[1] and mask.shape[2] == 1

        seq_len, batch_size, _ = query.shape
        
        # [seq_len_q, seq_len_k, batch_size]
        mask = self.get_mask(mask, query.shape, key.shape)

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        scores = self.get_scores(query, key)

        if self.alibi_biases is None or self.alibi_biases.shape[1] < seq_len:
            self.alibi_biases = get_alibi_biases(self.heads, mask[:, :, 0, 0])
        
        # Add ALiBi biases to scores
        # alibi_biases: [seq_len_q, seq_len_k, n_heads]
        scores = scores + self.alibi_biases[:seq_len, :seq_len, None, :]
        
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)

        attn = self.Dropout(attn)

        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)

        x = x.reshape(seq_len, batch_size, -1)

        return self.W_o(x)

def _test_alibi():
    # Set parameters
    d_model = 512
    heads = 8
    seq_len = 10
    batch_size = 2
    dropout = 0.1
    
    # Create model
    alibi_attention = ALiBiMultiHeadAttention(heads=heads, d_model=d_model, dropout=dropout)
    
    # Create input tensors
    query = torch.rand(seq_len, batch_size, d_model)
    key = torch.rand(seq_len, batch_size, d_model)
    value = torch.rand(seq_len, batch_size, d_model)
    
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len, 1)).bool()
    
    # Forward pass
    output = alibi_attention(query, key, value, mask)
    
    # Check output shape
    assert output.shape == (seq_len, batch_size, d_model), f"Output shape mismatch: got {output.shape}, expected {(seq_len, batch_size, d_model)}"
    
    print("ALiBi Multi-head attention test passed!")
    return output

if __name__ == "__main__":
    _test_alibi()
