import torch
import torch.nn as nn
import math
from MultiHeadAttention import MultiHeadAttention
from FFN import FFN
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int):
        super().__init__()
        self.linear = nn.Linear(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encoding = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(max_len,1,d_model), nonlinearity='relu'), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe


class TransformerLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 self_attn: MultiHeadAttention, 
                 src_attn: MultiHeadAttention,
                 ffn: FFN,
                 dropout: float):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
    
    def forward(self, 
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        
        #pre-norm better than post-norm
        z = self.norm_self_attn(x)
        self_attn_out = self.self_attn(query = z, key = z, value = z, mask = mask)

        x = x + self.dropout(self_attn_out)

        if src is not None:
            z = self.norm_src_attn(x)
            src_attn_out = self.src_attn(query = z, key = src, value = src, mask = src_mask)
            x = x + self.dropout(src_attn_out)
        
        z = self.norm_ff(x)
        
        ff = self.ffn(z)

        x = x + self.dropout(ff)

        return x

class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm([layer.d_model])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm([layer.d_model])
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_vocab)
    
    def forward(self, x: torch.Tensor):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: Generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        encode = self.encode(src, src_mask)
        
        return self.decode(encode, src_mask, tgt, tgt_mask)
        
