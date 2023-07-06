
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_fc1 = Linear(in_features=32, out_features=64, bias=True)
        self.load_state_dict(torch.load(r'fx_mlp/state_dict.pt'))

    
    
    def forward(self, inp : torch.Tensor):
        model_fc1 = self.model_fc1(inp);  inp = None
        gelu = torch._C._nn.gelu(model_fc1);  model_fc1 = None
        return (gelu,)
        
