
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
        self.load_state_dict(torch.load(r'mlp/state_dict.pt'))

    
    
    def forward(self, primals_1, primals_2, primals_3):
        t = torch.ops.aten.t.default(primals_1);  primals_1 = None
        addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = t = None
        gelu = torch.ops.aten.gelu.default(addmm)
        return [gelu, primals_3, addmm]
        
