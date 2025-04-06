import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation = nn.ReLU(),
                 bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

class StandardFFN(BaseFFN):
    def __init__(self, d_model: int, d_ff: int, **kwargs):
        super().__init__(d_model, d_ff, **kwargs)
        self.layer1 = nn.Linear(d_model, d_ff, bias=kwargs.get('bias', True))
        self.layer2 = nn.Linear(d_ff, d_model, bias=kwargs.get('bias', True))
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)

class GLU(BaseFFN):
    def __init__(self, d_model: int, d_ff: int, **kwargs):
        super().__init__(d_model, d_ff, **kwargs)
        self.linear = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        x = self.linear(x) * g
        x = self.dropout(x)
        return self.output(x)

class SwiGLU(BaseFFN):
    def __init__(self, d_model: int, d_ff: int, **kwargs):
        super().__init__(d_model, d_ff, **kwargs)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x1 = self.w1(x)
        x2 = torch.sigmoid(self.w2(x))
        x3 = self.w3(x)
        x = x1 * x2 * x3
        x = self.dropout(x)
        return self.output(x)



class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.5,
                 activation = nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=bias_gate)
    
    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            #gFFN
            x = g * self.linear_gate(x)
        else:
            x = g
        x = self.dropout(x)

        return self.layer2(x)
            
            
