import torch 
import torch.nn as nn
from Classic_Transformers.Transformer import Encoder, Decoder

from labml.configs import BaseConfigs, option, meta_config
from labml_nn.optimizers import WeightDecay
from typing import Tuple


class OptimizerConfigs(BaseConfigs):
    """
    <a id="OptimizerConfigs"></a>

    ## Optimizer Configurations
    """

    # Optimizer
    optimizer: torch.optim.Adam

    # Weight decay
    weight_decay_obj: WeightDecay
    # Whether weight decay is decoupled;
    # i.e. weight decay is not added to gradients
    weight_decouple: bool = True
    # Weight decay
    weight_decay: float = 0.0
    # Whether weight decay is absolute or should be multiplied by learning rate
    weight_decay_absolute: bool = False

    # Whether the adam update is optimized (different epsilon)
    optimized_adam_update: bool = True

    # Parameters to be optimized
    parameters: any

    # Learning rate $\alpha$
    learning_rate: float = 0.01
    # Beta values $(\beta_1, \beta_2)$ for Adam
    betas: Tuple[float, float] = (0.9, 0.999)
    # Epsilon $\epsilon$ for adam
    eps: float = 1e-08

    # Momentum for SGD
    momentum: float = 0.5
    # Whether to use AMSGrad
    amsgrad: bool = False

    # Number of warmup optimizer steps
    warmup: int = 2_000
    # Total number of optimizer steps (for cosine decay)
    total_steps: int = int(1e10)

    # Whether to degenerate to SGD in AdaBelief
    degenerate_to_sgd: bool = True

    # Whether to use Rectified Adam in AdaBelief
    rectify: bool = True

    # Model embedding size for Noam optimizer
    d_model: int

    rho: float

    def __init__(self):
        super().__init__(_primary='optimizer')


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

    
