from torchvision.models import resnet18

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.graph
import torch._inductor.utils
import torch.fx
import torch.nn as nn
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.scheduler import Scheduler
from torch._inductor.compile_fx import compile_fx


all_lowerings = {}


def patch_graph_lowering():

    old_init = torch._inductor.graph.GraphLowering.__init__

    def new_init(self, *args, **kwargs):
        print(f"Registering lowering {id(self)} with args {args} and kwargs {kwargs}")
        all_lowerings[id(self)] = self
        return old_init(self, *args, **kwargs)

    torch._inductor.graph.GraphLowering.__init__ = new_init


patch_graph_lowering()

# lowering = set(all_lowerings.values()).pop()
# scheduler: Scheduler = lowering.scheduler


def to_dict(obj):
    from torch._inductor.ir import (
        ExternKernelOut,
        FixedLayout,
        InputBuffer,
        ReinterpretView,
        StorageBox,
        ComputedBuffer,
    )

    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, ComputedBuffer):
        return dict(**{k: to_dict(v) for k, v in obj.__dict__.items() if k in ("name", "layout")}, **{"type": "ComputedBuffer"},)
    elif isinstance(
        obj,
        (
            ExternKernelOut,
            FixedLayout,
            InputBuffer,
            ReinterpretView,
            StorageBox,
        ),
    ):
        return dict(**{k: to_dict(v) for k, v in obj.__dict__.items()},**{"type": str(obj.__class__.__name__)})
    elif isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    else:
        return obj


def new_codegen(self: Scheduler):
    from torch._inductor.scheduler import NopKernelSchedulerNode, FusedSchedulerNode, SchedulerNode, V, config

    kaas_schedule = []

    for node in self.nodes:
        print(f"Codegen node {node}")
        self.buffer_names_no_longer_needed.update(node.last_usage)

        if not isinstance(node, NopKernelSchedulerNode):
            device = node.get_device()
            if device != self.current_device or node.is_extern() or node.is_template():
                self.flush()
            if device != self.current_device:
                if device.type == "cuda":
                    if self.current_device and self.current_device.type == "cuda":
                        V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                    assert device.index is not None, "device should have an index"
                    V.graph.wrapper_code.codegen_cuda_device_guard_enter(device.index)
                elif self.current_device and self.current_device.type == "cuda":
                    V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                self.current_device = device

        self.buffer_names_to_free.update(node.last_usage)

        if node.is_template():
            node, *epilogue = node.get_nodes()
            self.get_backend(device).codegen_template(node, epilogue)
        elif node.is_extern():
            kaas_schedule.append(
                {
                    "type": "extern",
                    "name": node.get_name(),
                    "kernel": node.node.kernel,
                    "buffer": to_dict(node.node),
                }
            )
            self.codegen_extern_call(node)
        elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
            self.get_backend(device).codegen_nodes(node.get_nodes())
            last_kv_pair = V.graph.wrapper_code.kernels.copy().popitem()
            kaas_schedule.append(
                {
                    "type": "triton",
                    "name": node.get_name(),
                    "triton_code_name": last_kv_pair[1],
                    "triton_code_body": last_kv_pair[0],
                    "buffer": to_dict(node.node),
                }
            )
        else:
            assert isinstance(node, NopKernelSchedulerNode)
            node.allocate()

        if config.triton.debug_sync_kernel:
            self.get_backend(device).codegen_sync()

        self.available_buffer_names.update(node.get_names())

    from pprint import pprint

    print("Kaas schedule:")
    for node in kaas_schedule:
        pprint(node)
    print("Graph info:")
    pprint(
        {
            "inputs": list(V.graph.graph_inputs.keys()),
            "outputs": [buf.data.name for buf in V.graph.graph_outputs],  # assume it's a StorageBox for now
        }
    )

    # From the Triton kernel code, you can run
    # inspect.getclosurevars(triton_.launchers[0]).globals["bin"].asm["ptx"] to get the compiled PTX code.
    # After JIT it though.
    # buf0 = empty_strided((8, 64), (64, 1), device='cuda', dtype=torch.float32)
    # stream0 = get_cuda_stream(0)
    # triton_.run(buf0, 512, grid=grid(512), stream=stream0)

    self.flush()


Scheduler.codegen = new_codegen


# dynamo_config.output_graph_code = True
# dynamo_config.output_code = True
# inductor_config.debug = True


torch.no_grad().__enter__()
torch.inference_mode().__enter__()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.relu(x)
        return x


model = MLP().eval().cuda()
inp = torch.randn(8, 32).cuda()

# model = resnet18(pretrained=True).eval().cuda()
# inp = torch.zeros((1, 3, 224, 224)).cuda()


res = None


def eager_optimizer(fx: torch.fx.graph_module.GraphModule, inp):
    return compile_fx(fx, inp)

    def my_compiler(fx, sample_inp):
        return compile_fx(fx, sample_inp)
        # fx.to_folder("resnet18-prims")
        # return fx.forward

    return aot_module_simplified(fx, inp, my_compiler, bw_compiler=lambda fx, _: fx.forward)


@torch.compile(backend=eager_optimizer)
def inference_func(inp):
    return model(inp)


image = inference_func(inp)
print("Got output", image.shape)
