from torch import nn
import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses import FakeTensorMode
from torch._dynamo.utils import fake_mode_from_tensors


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        return x


model = MLP().cuda()
inp = torch.randn(8, 32).cuda()


class MyInterpreter(torch.fx.Interpreter):
    def __init__(self, graph):
        super().__init__(graph)

    def run_node(self, n):
        if n.op == "call_function":
            if n.target == torch.nn.functional.gelu:
                print("GELU")
                return torch.nn.functional.gelu(n.args[0])
        return super().run_node(n)


import copy


def my_compiler(fx: torch.fx.GraphModule, sample_inp):
    fake_mode = fake_mode_from_tensors(sample_inp)
    with fake_mode:
        new_fx = copy.deepcopy(fx)
        ShapeProp(new_fx).propagate(*sample_inp)
        new_fx.graph.print_tabular()
    print(dict(fx.named_modules()))
    print(dict(fx.named_buffers()))
    print(dict(fx.named_parameters()))

    fx.to_folder("fx_mlp")
    interpreter = MyInterpreter(fx)
    interpreter.run(*sample_inp)
    return fx.forward


@torch.compile(backend=my_compiler)
def inference_func(inp):
    return model(inp)


res = inference_func(inp)
