from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_xmo/64/c64hdcytpd2ltw22ekovhipzqvvzmicz5fcgfwhf4zmwkwfpbrtu.py
triton_fused_gelu_relu_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tl.where(0 != 0, 0, tl.where(0 > tmp8, 0, tmp8))
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 64), (64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(arg1_1, arg2_1, as_strided(arg0_1, (32, 64), (1, 32)), alpha=1, beta=1, out=buf0)
        del arg0_1
        del arg1_1
        del arg2_1
        buf1 = buf0; del buf0  # reuse
        stream0 = get_cuda_stream(0)
        triton_fused_gelu_relu_0.run(buf1, 512, grid=grid(512), stream=stream0)
        return (buf1, )


def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))


if __name__ == "__main__":
    import argparse
    from torch._inductor.utils import benchmark_all_kernels

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels('None')
    else:
        benchmark_compiled_module()