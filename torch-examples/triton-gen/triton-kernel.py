import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor


@pointwise(
    size_hints=[512],
    filename=__file__,
    meta={
        "signature": {0: "*fp32", 1: "i32"},
        "device": 0,
        "constants": {},
        "mutated_arg_names": ["in_out_ptr0"],
        "configs": [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())],
    },
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
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
