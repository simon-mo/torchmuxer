
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
        self.load_state_dict(torch.load(r'mlp-prims/state_dict.pt'))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = arg2_1 = t = None
        gelu = torch.ops.aten.gelu.default(addmm);  addmm = None
        return (gelu,)
        
