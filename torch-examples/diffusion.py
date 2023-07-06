import inspect

import torch
import torch._dynamo as dynamo
import torch.fx
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from py_obj_scanner import _PyObjScanner
from contextlib import contextmanager
import collections.abc

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")


class ModuleFinder:
    def __init__(self) -> None:
        self.table = {}
        self.current_variable_stack = []
        self.scanned_ids = set()

    @contextmanager
    def set_stack(self, attr):
        self.current_variable_stack.append(attr)
        yield ".".join(self.current_variable_stack)
        self.current_variable_stack.pop()

    def find_nested(self, obj):
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                with self.set_stack(str(i)):
                    self.find_nested(item)
        elif isinstance(obj, collections.abc.Mapping):
            for i, item in obj.items():
                with self.set_stack(str(i)):
                    self.find_nested(item)
        elif not hasattr(obj, "__dict__"):
            # print(f"skipping {type(obj)}")
            return
        else:
            for attr in obj.__dict__:
                with self.set_stack(attr) as tree:
                    value = getattr(obj, attr)
                    if isinstance(value, torch.nn.Module):
                        for path, mod in value.named_modules():
                            self.table[id(mod)] = f"{tree}:{path}"
                    elif attr.startswith("__"):
                        continue
                    else:
                        # print(type(value))
                        if id(value) in self.scanned_ids:
                            continue
                        self.scanned_ids.add(id(value))
                        # print(tree)
                        self.find_nested(value)


finder = ModuleFinder()
finder.find_nested(pipe)
print(finder.table)


from torch._inductor.compile_fx import compile_fx


# def eager_optimizer(fx: torch.fx.graph_module.GraphModule, inp):
#     # print("compiling...")
#     # for mod in fx.children():
#     #     print(finder.table[id(mod)])
#     #     break
#     # print()


#     # for mod in fx.children():
#     #     if len(list(mod.children())) == 0 and id(mod) not in finder.table:
#     #         fx.graph.print_tabular()
#     #         import ipdb

#     #         ipdb.set_trace()
#     return compile_fx(fx, inp)

    # return fx.forward


# @torch.compile(backend=eager_optimizer)
@torch.compile(backend="inductor")
def inference_func(promt):
    image = pipe(prompt, num_inference_steps=1).images[0]
    return image


from torch.utils._python_dispatch import TorchDispatchMode


class PrintingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"{func.__module__}.{func.__name__}")
        return func(*args, **kwargs)


prompt = "a photo of an astronaut riding a horse on mars"
image = inference_func(prompt)
# with PrintingMode():
# image = inference_func(prompt)

# (
#     explanation,
#     out_guards,
#     graphs,
#     ops_per_graph,
#     break_reasons,
#     explanation_verbose,
# ) = dynamo.explain(inference_func, [prompt])
# print(explanation)
# import ipdb

# ipdb.set_trace()


# image.save("astronaut_rides_horse.png")
