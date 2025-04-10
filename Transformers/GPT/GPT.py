import torch 
import torch.nn as nn
from labml_nn.optimizers.configs import OptimizerConfigs
from Classic_Transformers.Transformer import Encoder, Decoder


def subsequent_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    return mask

class GPT(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator
        self.mask = None
    

    def forward(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != len(x):
            self.mask = subsequent_mask(len(x)).to(x.device)
        
        x = self.src_embed(x)

        x = self.encoder(x, self.mask)

        x =self.generator(x)

        return x


