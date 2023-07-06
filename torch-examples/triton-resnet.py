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
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton__0 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
"""
)


triton__1 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
"""
)


triton__2 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (6272*x3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 - tmp1
        tmp3 = tmp2 * tmp2
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp4, xmask)
"""
)


triton__3 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0)
    tmp13 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = 1.0000797257434426
    tmp10 = tmp5 * tmp9
    tmp11 = 0.1
    tmp12 = tmp10 * tmp11
    tmp14 = 0.9
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp8, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
    tl.store(out_ptr0 + x0, tmp3, xmask)
"""
)


triton__4 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
"""
)


triton__5 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 56)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 112
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 > tmp12, tmp19, tmp12))
    tmp21 = 1 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = tl.where(tmp27 != tmp27, tmp27, tl.where(tmp27 > tmp20, tmp27, tmp20))
    tmp29 = 2*x1
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = tl.where(tmp35 != tmp35, tmp35, tl.where(tmp35 > tmp28, tmp35, tmp28))
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + ((2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = tl.where(tmp39 != tmp39, tmp39, tl.where(tmp39 > tmp36, tmp39, tmp36))
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = tl.where(tmp43 != tmp43, tmp43, tl.where(tmp43 > tmp40, tmp43, tmp40))
    tmp45 = 1 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = tl.where(tmp51 != tmp51, tmp51, tl.where(tmp51 > tmp44, tmp51, tmp44))
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = tl.where(tmp55 != tmp55, tmp55, tl.where(tmp55 > tmp52, tmp55, tmp52))
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 > tmp56, tmp59, tmp56))
    tmp61 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)
    tmp62 = tl.where(tmp10, tmp61, float("-inf"))
    tmp63 = (-113) + (2*x0) + (224*x1)
    tmp64 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0)
    tmp65 = tl.where(tmp17, tmp64, float("-inf"))
    tmp66 = (-112) + (2*x0) + (224*x1)
    tmp67 = tmp65 > tmp62
    tmp68 = tl.where(tmp67, tmp66, tmp63)
    tmp69 = tl.where(tmp65 != tmp65, tmp65, tl.where(tmp65 > tmp62, tmp65, tmp62))
    tmp70 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0)
    tmp71 = tl.where(tmp25, tmp70, float("-inf"))
    tmp72 = (-111) + (2*x0) + (224*x1)
    tmp73 = tmp71 > tmp69
    tmp74 = tl.where(tmp73, tmp72, tmp68)
    tmp75 = tl.where(tmp71 != tmp71, tmp71, tl.where(tmp71 > tmp69, tmp71, tmp69))
    tmp76 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0)
    tmp77 = tl.where(tmp33, tmp76, float("-inf"))
    tmp78 = (-1) + (2*x0) + (224*x1)
    tmp79 = tmp77 > tmp75
    tmp80 = tl.where(tmp79, tmp78, tmp74)
    tmp81 = tl.where(tmp77 != tmp77, tmp77, tl.where(tmp77 > tmp75, tmp77, tmp75))
    tmp82 = tl.load(in_ptr0 + ((2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0)
    tmp83 = tl.where(tmp37, tmp82, float("-inf"))
    tmp84 = (2*x0) + (224*x1)
    tmp85 = tmp83 > tmp81
    tmp86 = tl.where(tmp85, tmp84, tmp80)
    tmp87 = tl.where(tmp83 != tmp83, tmp83, tl.where(tmp83 > tmp81, tmp83, tmp81))
    tmp88 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0)
    tmp89 = tl.where(tmp41, tmp88, float("-inf"))
    tmp90 = 1 + (2*x0) + (224*x1)
    tmp91 = tmp89 > tmp87
    tmp92 = tl.where(tmp91, tmp90, tmp86)
    tmp93 = tl.where(tmp89 != tmp89, tmp89, tl.where(tmp89 > tmp87, tmp89, tmp87))
    tmp94 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0)
    tmp95 = tl.where(tmp49, tmp94, float("-inf"))
    tmp96 = 111 + (2*x0) + (224*x1)
    tmp97 = tmp95 > tmp93
    tmp98 = tl.where(tmp97, tmp96, tmp92)
    tmp99 = tl.where(tmp95 != tmp95, tmp95, tl.where(tmp95 > tmp93, tmp95, tmp93))
    tmp100 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0)
    tmp101 = tl.where(tmp53, tmp100, float("-inf"))
    tmp102 = 112 + (2*x0) + (224*x1)
    tmp103 = tmp101 > tmp99
    tmp104 = tl.where(tmp103, tmp102, tmp98)
    tmp105 = tl.where(tmp101 != tmp101, tmp101, tl.where(tmp101 > tmp99, tmp101, tmp99))
    tmp106 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0)
    tmp107 = tl.where(tmp57, tmp106, float("-inf"))
    tmp108 = 113 + (2*x0) + (224*x1)
    tmp109 = tmp107 > tmp105
    tmp110 = tl.where(tmp109, tmp108, tmp104)
    tmp111 = tl.where(tmp107 != tmp107, tmp107, tl.where(tmp107 > tmp105, tmp107, tmp105))
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
    tl.store(out_ptr1 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp110, xmask)
"""
)


triton__6 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 3136.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp15 = tmp14 - tmp3
        tmp16 = 3136.0
        tmp17 = tmp13 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 + tmp24
        tmp26 = tl.where(0 != 0, 0, tl.where(0 > tmp25, 0, tmp25))
        tl.store(out_ptr2 + (r1 + (3136*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, rmask & xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp27 = 3136.0
    tmp28 = tmp13 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.libdevice.rsqrt(tmp30)
    tmp32 = 1.0003189792663476
    tmp33 = tmp28 * tmp32
    tmp34 = 0.1
    tmp35 = tmp33 * tmp34
    tmp37 = 0.9
    tmp38 = tmp36 * tmp37
    tmp39 = tmp35 + tmp38
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp31, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp39, xmask)
"""
)


triton__7 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 3136.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp26 = tl.load(in_ptr4 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp15 = tmp14 - tmp3
        tmp16 = 3136.0
        tmp17 = tmp13 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 + tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = tl.where(0 != 0, 0, tl.where(0 > tmp27, 0, tmp27))
        tl.store(out_ptr2 + (r1 + (3136*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask & xmask)
    tmp38 = tl.load(in_ptr5 + (x0), xmask)
    tmp29 = 3136.0
    tmp30 = tmp13 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = tl.libdevice.rsqrt(tmp32)
    tmp34 = 1.0003189792663476
    tmp35 = tmp30 * tmp34
    tmp36 = 0.1
    tmp37 = tmp35 * tmp36
    tmp39 = 0.9
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 + tmp40
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp33, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp41, xmask)
"""
)


triton__8 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    tmp30 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 784.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 - tmp5
    tmp13 = tmp12 * tmp12
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(0 != 0, 0, tl.where(0 > tmp25, 0, tmp25))
    tmp27 = 1.0012771392081736
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp6
    tmp31 = tmp30 * tmp9
    tmp32 = tmp29 + tmp31
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(out_ptr2 + (r1 + (784*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp32, xmask)
"""
)


triton__9 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32', 20: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp12 = tl.load(in_ptr2 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp18 = tl.load(in_ptr3 + (x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp38 = tl.load(in_ptr5 + (x0), xmask)
    tmp44 = tl.load(in_ptr6 + (x0), xmask)
    tmp46 = tl.load(in_ptr7 + (x0), xmask)
    tmp53 = tl.load(in_ptr8 + (x0), xmask)
    tmp58 = tl.load(in_ptr9 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 784.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp15 / tmp4
    tmp17 = tmp16 * tmp6
    tmp19 = tmp18 * tmp9
    tmp20 = tmp17 + tmp19
    tmp21 = tmp0 - tmp5
    tmp22 = tmp21 * tmp21
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp12 - tmp16
    tmp27 = tmp26 * tmp26
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp25 / tmp4
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.libdevice.rsqrt(tmp33)
    tmp35 = tmp21 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp30 / tmp4
    tmp41 = tmp40 + tmp32
    tmp42 = tl.libdevice.rsqrt(tmp41)
    tmp43 = tmp26 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp39 + tmp47
    tmp49 = tl.where(0 != 0, 0, tl.where(0 > tmp48, 0, tmp48))
    tmp50 = 1.0012771392081736
    tmp51 = tmp31 * tmp50
    tmp52 = tmp51 * tmp6
    tmp54 = tmp53 * tmp9
    tmp55 = tmp52 + tmp54
    tmp56 = tmp40 * tmp50
    tmp57 = tmp56 * tmp6
    tmp59 = tmp58 * tmp9
    tmp60 = tmp57 + tmp59
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(in_out_ptr2 + (r1 + (784*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp55, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp42, xmask)
    tl.store(out_ptr7 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp60, xmask)
"""
)


triton__10 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    tmp26 = tl.load(in_ptr4 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp32 = tl.load(in_ptr5 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 784.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 - tmp5
    tmp13 = tmp12 * tmp12
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tl.where(0 != 0, 0, tl.where(0 > tmp27, 0, tmp27))
    tmp29 = 1.0012771392081736
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 * tmp6
    tmp33 = tmp32 * tmp9
    tmp34 = tmp31 + tmp33
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(out_ptr2 + (r1 + (784*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask & xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
"""
)


triton__11 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    tmp30 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 196.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 - tmp5
    tmp13 = tmp12 * tmp12
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(0 != 0, 0, tl.where(0 > tmp25, 0, tmp25))
    tmp27 = 1.005128205128205
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp6
    tmp31 = tmp30 * tmp9
    tmp32 = tmp29 + tmp31
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(out_ptr2 + (r1 + (196*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp32, xmask)
"""
)


triton__12 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32', 20: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp12 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask & xmask, other=0)
    tmp18 = tl.load(in_ptr3 + (x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp38 = tl.load(in_ptr5 + (x0), xmask)
    tmp44 = tl.load(in_ptr6 + (x0), xmask)
    tmp46 = tl.load(in_ptr7 + (x0), xmask)
    tmp53 = tl.load(in_ptr8 + (x0), xmask)
    tmp58 = tl.load(in_ptr9 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 196.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp15 / tmp4
    tmp17 = tmp16 * tmp6
    tmp19 = tmp18 * tmp9
    tmp20 = tmp17 + tmp19
    tmp21 = tmp0 - tmp5
    tmp22 = tmp21 * tmp21
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp12 - tmp16
    tmp27 = tmp26 * tmp26
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp25 / tmp4
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.libdevice.rsqrt(tmp33)
    tmp35 = tmp21 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp30 / tmp4
    tmp41 = tmp40 + tmp32
    tmp42 = tl.libdevice.rsqrt(tmp41)
    tmp43 = tmp26 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp39 + tmp47
    tmp49 = tl.where(0 != 0, 0, tl.where(0 > tmp48, 0, tmp48))
    tmp50 = 1.005128205128205
    tmp51 = tmp31 * tmp50
    tmp52 = tmp51 * tmp6
    tmp54 = tmp53 * tmp9
    tmp55 = tmp52 + tmp54
    tmp56 = tmp40 * tmp50
    tmp57 = tmp56 * tmp6
    tmp59 = tmp58 * tmp9
    tmp60 = tmp57 + tmp59
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(in_out_ptr2 + (r1 + (196*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp55, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp42, xmask)
    tl.store(out_ptr7 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp60, xmask)
"""
)


triton__13 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    tmp26 = tl.load(in_ptr4 + (r1 + (196*x0)), rmask & xmask, other=0)
    tmp32 = tl.load(in_ptr5 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 196.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 - tmp5
    tmp13 = tmp12 * tmp12
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tl.where(0 != 0, 0, tl.where(0 > tmp27, 0, tmp27))
    tmp29 = 1.005128205128205
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 * tmp6
    tmp33 = tmp32 * tmp9
    tmp34 = tmp31 + tmp33
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(out_ptr2 + (r1 + (196*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask & xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
"""
)


triton__14 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    tmp30 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 49.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 - tmp5
    tmp13 = tmp12 * tmp12
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(0 != 0, 0, tl.where(0 > tmp25, 0, tmp25))
    tmp27 = 1.0208333333333333
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp6
    tmp31 = tmp30 * tmp9
    tmp32 = tmp29 + tmp31
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(out_ptr2 + (r1 + (49*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp32, xmask)
"""
)


triton__15 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32', 20: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp12 = tl.load(in_ptr2 + (r1 + (49*x0)), rmask & xmask, other=0)
    tmp18 = tl.load(in_ptr3 + (x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp38 = tl.load(in_ptr5 + (x0), xmask)
    tmp44 = tl.load(in_ptr6 + (x0), xmask)
    tmp46 = tl.load(in_ptr7 + (x0), xmask)
    tmp53 = tl.load(in_ptr8 + (x0), xmask)
    tmp58 = tl.load(in_ptr9 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 49.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp15 / tmp4
    tmp17 = tmp16 * tmp6
    tmp19 = tmp18 * tmp9
    tmp20 = tmp17 + tmp19
    tmp21 = tmp0 - tmp5
    tmp22 = tmp21 * tmp21
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp12 - tmp16
    tmp27 = tmp26 * tmp26
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp25 / tmp4
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.libdevice.rsqrt(tmp33)
    tmp35 = tmp21 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp30 / tmp4
    tmp41 = tmp40 + tmp32
    tmp42 = tl.libdevice.rsqrt(tmp41)
    tmp43 = tmp26 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp39 + tmp47
    tmp49 = tl.where(0 != 0, 0, tl.where(0 > tmp48, 0, tmp48))
    tmp50 = 1.0208333333333333
    tmp51 = tmp31 * tmp50
    tmp52 = tmp51 * tmp6
    tmp54 = tmp53 * tmp9
    tmp55 = tmp52 + tmp54
    tmp56 = tmp40 * tmp50
    tmp57 = tmp56 * tmp6
    tmp59 = tmp58 * tmp9
    tmp60 = tmp57 + tmp59
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(in_out_ptr2 + (r1 + (49*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp55, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp42, xmask)
    tl.store(out_ptr7 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp60, xmask)
"""
)


triton__16 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp22 = tl.load(in_ptr2 + (x0), xmask)
    tmp24 = tl.load(in_ptr3 + (x0), xmask)
    tmp26 = tl.load(in_ptr4 + (r1 + (49*x0)), rmask & xmask, other=0)
    tmp37 = tl.load(in_ptr5 + (x0), xmask)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tmp4 = 49.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp9 = 0.9
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 - tmp5
    tmp13 = tmp12 * tmp12
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tl.where(0 != 0, 0, tl.where(0 > tmp27, 0, tmp27))
    tmp29 = 0.0
    tmp30 = tmp28 <= tmp29
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 1.0208333333333333
    tmp35 = tmp17 * tmp34
    tmp36 = tmp35 * tmp6
    tmp38 = tmp37 * tmp9
    tmp39 = tmp36 + tmp38
    tmp40 = tmp33 / tmp4
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
    tl.store(out_ptr3 + (r1 + (49*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp30, rmask & xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp39, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp40, xmask)
"""
)


triton__17 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK])
    tmp1 = 1
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp2, None)
"""
)


async_compile.wait(globals())
del async_compile


def call(args):
    (
        primals_1,
        primals_2,
        primals_3,
        primals_4,
        primals_5,
        primals_6,
        primals_7,
        primals_8,
        primals_9,
        primals_10,
        primals_11,
        primals_12,
        primals_13,
        primals_14,
        primals_15,
        primals_16,
        primals_17,
        primals_18,
        primals_19,
        primals_20,
        primals_21,
        primals_22,
        primals_23,
        primals_24,
        primals_25,
        primals_26,
        primals_27,
        primals_28,
        primals_29,
        primals_30,
        primals_31,
        primals_32,
        primals_33,
        primals_34,
        primals_35,
        primals_36,
        primals_37,
        primals_38,
        primals_39,
        primals_40,
        primals_41,
        primals_42,
        primals_43,
        primals_44,
        primals_45,
        primals_46,
        primals_47,
        primals_48,
        primals_49,
        primals_50,
        primals_51,
        primals_52,
        primals_53,
        primals_54,
        primals_55,
        primals_56,
        primals_57,
        primals_58,
        primals_59,
        primals_60,
        primals_61,
        primals_62,
        primals_63,
        primals_64,
        primals_65,
        primals_66,
        primals_67,
        primals_68,
        primals_69,
        primals_70,
        primals_71,
        primals_72,
        primals_73,
        primals_74,
        primals_75,
        primals_76,
        primals_77,
        primals_78,
        primals_79,
        primals_80,
        primals_81,
        primals_82,
        primals_83,
        primals_84,
        primals_85,
        primals_86,
        primals_87,
        primals_88,
        primals_89,
        primals_90,
        primals_91,
        primals_92,
        primals_93,
        primals_94,
        primals_95,
        primals_96,
        primals_97,
        primals_98,
        primals_99,
        primals_100,
        primals_101,
        primals_102,
        primals_103,
        primals_104,
        primals_105,
        primals_106,
        primals_107,
        primals_108,
        primals_109,
        primals_110,
        primals_111,
        primals_112,
        primals_113,
        primals_114,
        primals_115,
        primals_116,
        primals_117,
        primals_118,
        primals_119,
        primals_120,
        primals_121,
        primals_122,
        primals_123,
    ) = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = aten.convolution(
            primals_123, primals_1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf0, (1, 64, 112, 112), (802816, 12544, 112, 1))
        buf1 = empty_strided(
            (1, 64, 1, 1, 2), (128, 2, 128, 128, 1), device="cuda", dtype=torch.float32
        )
        stream0 = get_cuda_stream(0)
        triton__0.run(buf0, buf1, 128, 6272, grid=grid(128), stream=stream0)
        buf2 = empty_strided(
            (1, 64, 1, 1), (64, 1, 64, 64), device="cuda", dtype=torch.float32
        )
        buf3 = buf2
        del buf2  # reuse
        buf7 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        triton__1.run(
            buf3, buf1, primals_63, buf7, 64, 2, grid=grid(64), stream=stream0
        )
        del primals_63
        buf4 = buf1
        del buf1  # reuse
        triton__2.run(buf0, buf3, buf4, 128, 6272, grid=grid(128), stream=stream0)
        buf5 = empty_strided(
            (1, 64, 1, 1), (64, 1, 64, 64), device="cuda", dtype=torch.float32
        )
        buf6 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf8 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        triton__3.run(
            buf4, primals_64, buf5, buf6, buf8, 64, 2, grid=grid(64), stream=stream0
        )
        del primals_64
        buf9 = empty_strided(
            (1, 64, 112, 112),
            (802816, 12544, 112, 1),
            device="cuda",
            dtype=torch.float32,
        )
        triton__4.run(
            buf0,
            buf3,
            buf5,
            primals_2,
            primals_3,
            buf9,
            802816,
            grid=grid(802816),
            stream=stream0,
        )
        del primals_3
        buf10 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.float32
        )
        buf11 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.int64
        )
        triton__5.run(buf9, buf10, buf11, 200704, grid=grid(200704), stream=stream0)
        buf12 = aten.convolution(
            buf10, primals_4, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf12, (1, 64, 56, 56), (200704, 3136, 56, 1))
        buf13 = buf5
        del buf5  # reuse
        buf14 = buf13
        del buf13  # reuse
        buf17 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf19 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.float32
        )
        buf16 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf18 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        triton__6.run(
            buf14,
            buf12,
            primals_66,
            primals_5,
            primals_6,
            primals_67,
            buf17,
            buf19,
            buf16,
            buf18,
            64,
            3136,
            grid=grid(64),
            stream=stream0,
        )
        del primals_6
        del primals_66
        del primals_67
        buf20 = aten.convolution(
            buf19, primals_7, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf20, (1, 64, 56, 56), (200704, 3136, 56, 1))
        buf21 = empty_strided(
            (1, 64, 1, 1), (64, 1, 64, 64), device="cuda", dtype=torch.float32
        )
        buf22 = buf21
        del buf21  # reuse
        buf25 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf27 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.float32
        )
        buf24 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf26 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        triton__7.run(
            buf22,
            buf20,
            primals_69,
            primals_8,
            primals_9,
            buf10,
            primals_70,
            buf25,
            buf27,
            buf24,
            buf26,
            64,
            3136,
            grid=grid(64),
            stream=stream0,
        )
        del primals_69
        del primals_70
        del primals_9
        buf28 = aten.convolution(
            buf27, primals_10, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf28, (1, 64, 56, 56), (200704, 3136, 56, 1))
        buf29 = empty_strided(
            (1, 64, 1, 1), (64, 1, 64, 64), device="cuda", dtype=torch.float32
        )
        buf30 = buf29
        del buf29  # reuse
        buf33 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf35 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.float32
        )
        buf32 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf34 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        triton__6.run(
            buf30,
            buf28,
            primals_72,
            primals_11,
            primals_12,
            primals_73,
            buf33,
            buf35,
            buf32,
            buf34,
            64,
            3136,
            grid=grid(64),
            stream=stream0,
        )
        del primals_12
        del primals_72
        del primals_73
        buf36 = aten.convolution(
            buf35, primals_13, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf36, (1, 64, 56, 56), (200704, 3136, 56, 1))
        buf37 = empty_strided(
            (1, 64, 1, 1), (64, 1, 64, 64), device="cuda", dtype=torch.float32
        )
        buf38 = buf37
        del buf37  # reuse
        buf41 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf43 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.float32
        )
        buf40 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        buf42 = empty_strided((64,), (1,), device="cuda", dtype=torch.float32)
        triton__7.run(
            buf38,
            buf36,
            primals_75,
            primals_14,
            primals_15,
            buf27,
            primals_76,
            buf41,
            buf43,
            buf40,
            buf42,
            64,
            3136,
            grid=grid(64),
            stream=stream0,
        )
        del primals_15
        del primals_75
        del primals_76
        buf44 = aten.convolution(
            buf43, primals_16, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf44, (1, 128, 28, 28), (100352, 784, 28, 1))
        buf45 = as_strided(buf4, (1, 128, 1, 1), (128, 1, 128, 128))
        del buf4  # reuse
        buf46 = buf45
        del buf45  # reuse
        buf49 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf51 = empty_strided(
            (1, 128, 28, 28), (100352, 784, 28, 1), device="cuda", dtype=torch.float32
        )
        buf48 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf50 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        triton__8.run(
            buf46,
            buf44,
            primals_78,
            primals_17,
            primals_18,
            primals_79,
            buf49,
            buf51,
            buf48,
            buf50,
            128,
            784,
            grid=grid(128),
            stream=stream0,
        )
        del primals_18
        del primals_78
        del primals_79
        buf52 = aten.convolution(
            buf51, primals_19, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf52, (1, 128, 28, 28), (100352, 784, 28, 1))
        buf59 = aten.convolution(
            buf43, primals_22, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf59, (1, 128, 28, 28), (100352, 784, 28, 1))
        buf53 = empty_strided(
            (1, 128, 1, 1), (128, 1, 128, 128), device="cuda", dtype=torch.float32
        )
        buf54 = buf53
        del buf53  # reuse
        buf57 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf60 = empty_strided(
            (1, 128, 1, 1), (128, 1, 128, 128), device="cuda", dtype=torch.float32
        )
        buf61 = buf60
        del buf60  # reuse
        buf64 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf66 = empty_strided(
            (1, 128, 28, 28), (100352, 784, 28, 1), device="cuda", dtype=torch.float32
        )
        buf67 = buf66
        del buf66  # reuse
        buf56 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf58 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf63 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf65 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        triton__9.run(
            buf54,
            buf61,
            buf67,
            buf52,
            primals_81,
            buf59,
            primals_84,
            primals_20,
            primals_21,
            primals_23,
            primals_24,
            primals_82,
            primals_85,
            buf57,
            buf64,
            buf56,
            buf58,
            buf63,
            buf65,
            128,
            784,
            grid=grid(128),
            stream=stream0,
        )
        del primals_21
        del primals_24
        del primals_81
        del primals_82
        del primals_84
        del primals_85
        buf68 = aten.convolution(
            buf67, primals_25, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf68, (1, 128, 28, 28), (100352, 784, 28, 1))
        buf69 = empty_strided(
            (1, 128, 1, 1), (128, 1, 128, 128), device="cuda", dtype=torch.float32
        )
        buf70 = buf69
        del buf69  # reuse
        buf73 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf75 = empty_strided(
            (1, 128, 28, 28), (100352, 784, 28, 1), device="cuda", dtype=torch.float32
        )
        buf72 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf74 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        triton__8.run(
            buf70,
            buf68,
            primals_87,
            primals_26,
            primals_27,
            primals_88,
            buf73,
            buf75,
            buf72,
            buf74,
            128,
            784,
            grid=grid(128),
            stream=stream0,
        )
        del primals_27
        del primals_87
        del primals_88
        buf76 = aten.convolution(
            buf75, primals_28, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf76, (1, 128, 28, 28), (100352, 784, 28, 1))
        buf77 = empty_strided(
            (1, 128, 1, 1), (128, 1, 128, 128), device="cuda", dtype=torch.float32
        )
        buf78 = buf77
        del buf77  # reuse
        buf81 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf83 = empty_strided(
            (1, 128, 28, 28), (100352, 784, 28, 1), device="cuda", dtype=torch.float32
        )
        buf80 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        buf82 = empty_strided((128,), (1,), device="cuda", dtype=torch.float32)
        triton__10.run(
            buf78,
            buf76,
            primals_90,
            primals_29,
            primals_30,
            buf67,
            primals_91,
            buf81,
            buf83,
            buf80,
            buf82,
            128,
            784,
            grid=grid(128),
            stream=stream0,
        )
        del primals_30
        del primals_90
        del primals_91
        buf84 = aten.convolution(
            buf83, primals_31, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf84, (1, 256, 14, 14), (50176, 196, 14, 1))
        buf85 = empty_strided(
            (1, 256, 1, 1), (256, 1, 256, 256), device="cuda", dtype=torch.float32
        )
        buf86 = buf85
        del buf85  # reuse
        buf89 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf91 = empty_strided(
            (1, 256, 14, 14), (50176, 196, 14, 1), device="cuda", dtype=torch.float32
        )
        buf88 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf90 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        triton__11.run(
            buf86,
            buf84,
            primals_93,
            primals_32,
            primals_33,
            primals_94,
            buf89,
            buf91,
            buf88,
            buf90,
            256,
            196,
            grid=grid(256),
            stream=stream0,
        )
        del primals_33
        del primals_93
        del primals_94
        buf92 = aten.convolution(
            buf91, primals_34, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf92, (1, 256, 14, 14), (50176, 196, 14, 1))
        buf99 = aten.convolution(
            buf83, primals_37, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf99, (1, 256, 14, 14), (50176, 196, 14, 1))
        buf93 = empty_strided(
            (1, 256, 1, 1), (256, 1, 256, 256), device="cuda", dtype=torch.float32
        )
        buf94 = buf93
        del buf93  # reuse
        buf97 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf100 = empty_strided(
            (1, 256, 1, 1), (256, 1, 256, 256), device="cuda", dtype=torch.float32
        )
        buf101 = buf100
        del buf100  # reuse
        buf104 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf106 = empty_strided(
            (1, 256, 14, 14), (50176, 196, 14, 1), device="cuda", dtype=torch.float32
        )
        buf107 = buf106
        del buf106  # reuse
        buf96 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf98 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf103 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf105 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        triton__12.run(
            buf94,
            buf101,
            buf107,
            buf92,
            primals_96,
            buf99,
            primals_99,
            primals_35,
            primals_36,
            primals_38,
            primals_39,
            primals_97,
            primals_100,
            buf97,
            buf104,
            buf96,
            buf98,
            buf103,
            buf105,
            256,
            196,
            grid=grid(256),
            stream=stream0,
        )
        del primals_100
        del primals_36
        del primals_39
        del primals_96
        del primals_97
        del primals_99
        buf108 = aten.convolution(
            buf107, primals_40, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf108, (1, 256, 14, 14), (50176, 196, 14, 1))
        buf109 = empty_strided(
            (1, 256, 1, 1), (256, 1, 256, 256), device="cuda", dtype=torch.float32
        )
        buf110 = buf109
        del buf109  # reuse
        buf113 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf115 = empty_strided(
            (1, 256, 14, 14), (50176, 196, 14, 1), device="cuda", dtype=torch.float32
        )
        buf112 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf114 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        triton__11.run(
            buf110,
            buf108,
            primals_102,
            primals_41,
            primals_42,
            primals_103,
            buf113,
            buf115,
            buf112,
            buf114,
            256,
            196,
            grid=grid(256),
            stream=stream0,
        )
        del primals_102
        del primals_103
        del primals_42
        buf116 = aten.convolution(
            buf115, primals_43, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf116, (1, 256, 14, 14), (50176, 196, 14, 1))
        buf117 = empty_strided(
            (1, 256, 1, 1), (256, 1, 256, 256), device="cuda", dtype=torch.float32
        )
        buf118 = buf117
        del buf117  # reuse
        buf121 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf123 = empty_strided(
            (1, 256, 14, 14), (50176, 196, 14, 1), device="cuda", dtype=torch.float32
        )
        buf120 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        buf122 = empty_strided((256,), (1,), device="cuda", dtype=torch.float32)
        triton__13.run(
            buf118,
            buf116,
            primals_105,
            primals_44,
            primals_45,
            buf107,
            primals_106,
            buf121,
            buf123,
            buf120,
            buf122,
            256,
            196,
            grid=grid(256),
            stream=stream0,
        )
        del primals_105
        del primals_106
        del primals_45
        buf124 = aten.convolution(
            buf123, primals_46, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf124, (1, 512, 7, 7), (25088, 49, 7, 1))
        buf125 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf126 = buf125
        del buf125  # reuse
        buf129 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf131 = empty_strided(
            (1, 512, 7, 7), (25088, 49, 7, 1), device="cuda", dtype=torch.float32
        )
        buf128 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf130 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        triton__14.run(
            buf126,
            buf124,
            primals_108,
            primals_47,
            primals_48,
            primals_109,
            buf129,
            buf131,
            buf128,
            buf130,
            512,
            49,
            grid=grid(512),
            stream=stream0,
        )
        del primals_108
        del primals_109
        del primals_48
        buf132 = aten.convolution(
            buf131, primals_49, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf132, (1, 512, 7, 7), (25088, 49, 7, 1))
        buf139 = aten.convolution(
            buf123, primals_52, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf139, (1, 512, 7, 7), (25088, 49, 7, 1))
        buf133 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf134 = buf133
        del buf133  # reuse
        buf137 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf140 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf141 = buf140
        del buf140  # reuse
        buf144 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf146 = empty_strided(
            (1, 512, 7, 7), (25088, 49, 7, 1), device="cuda", dtype=torch.float32
        )
        buf147 = buf146
        del buf146  # reuse
        buf136 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf138 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf143 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf145 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        triton__15.run(
            buf134,
            buf141,
            buf147,
            buf132,
            primals_111,
            buf139,
            primals_114,
            primals_50,
            primals_51,
            primals_53,
            primals_54,
            primals_112,
            primals_115,
            buf137,
            buf144,
            buf136,
            buf138,
            buf143,
            buf145,
            512,
            49,
            grid=grid(512),
            stream=stream0,
        )
        del primals_111
        del primals_112
        del primals_114
        del primals_115
        del primals_51
        del primals_54
        buf148 = aten.convolution(
            buf147, primals_55, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf148, (1, 512, 7, 7), (25088, 49, 7, 1))
        buf149 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf150 = buf149
        del buf149  # reuse
        buf153 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf155 = empty_strided(
            (1, 512, 7, 7), (25088, 49, 7, 1), device="cuda", dtype=torch.float32
        )
        buf152 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf154 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        triton__14.run(
            buf150,
            buf148,
            primals_117,
            primals_56,
            primals_57,
            primals_118,
            buf153,
            buf155,
            buf152,
            buf154,
            512,
            49,
            grid=grid(512),
            stream=stream0,
        )
        del primals_117
        del primals_118
        del primals_57
        buf156 = aten.convolution(
            buf155, primals_58, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf156, (1, 512, 7, 7), (25088, 49, 7, 1))
        buf157 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf158 = buf157
        del buf157  # reuse
        buf161 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf167 = empty_strided(
            (1, 512, 7, 7), (25088, 49, 7, 1), device="cuda", dtype=torch.bool
        )
        buf164 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf160 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf162 = empty_strided((512,), (1,), device="cuda", dtype=torch.float32)
        buf165 = as_strided(buf164, (1, 512), (512, 1))
        del buf164  # reuse
        triton__16.run(
            buf158,
            buf165,
            buf156,
            primals_120,
            primals_59,
            primals_60,
            buf147,
            primals_121,
            buf161,
            buf167,
            buf160,
            buf162,
            512,
            49,
            grid=grid(512),
            stream=stream0,
        )
        del primals_120
        del primals_121
        del primals_60
        buf166 = empty_strided((1, 1000), (1000, 1), device="cuda", dtype=torch.float32)
        extern_kernels.addmm(
            primals_62,
            buf165,
            as_strided(primals_61, (512, 1000), (1, 512)),
            alpha=1,
            beta=1,
            out=buf166,
        )
        del primals_62
        buf168 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_65, buf168, 1, grid=grid(1), stream=stream0)
        del primals_65
        buf169 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_68, buf169, 1, grid=grid(1), stream=stream0)
        del primals_68
        buf170 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_71, buf170, 1, grid=grid(1), stream=stream0)
        del primals_71
        buf171 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_74, buf171, 1, grid=grid(1), stream=stream0)
        del primals_74
        buf172 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_77, buf172, 1, grid=grid(1), stream=stream0)
        del primals_77
        buf173 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_80, buf173, 1, grid=grid(1), stream=stream0)
        del primals_80
        buf174 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_83, buf174, 1, grid=grid(1), stream=stream0)
        del primals_83
        buf175 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_86, buf175, 1, grid=grid(1), stream=stream0)
        del primals_86
        buf176 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_89, buf176, 1, grid=grid(1), stream=stream0)
        del primals_89
        buf177 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_92, buf177, 1, grid=grid(1), stream=stream0)
        del primals_92
        buf178 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_95, buf178, 1, grid=grid(1), stream=stream0)
        del primals_95
        buf179 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_98, buf179, 1, grid=grid(1), stream=stream0)
        del primals_98
        buf180 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_101, buf180, 1, grid=grid(1), stream=stream0)
        del primals_101
        buf181 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_104, buf181, 1, grid=grid(1), stream=stream0)
        del primals_104
        buf182 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_107, buf182, 1, grid=grid(1), stream=stream0)
        del primals_107
        buf183 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_110, buf183, 1, grid=grid(1), stream=stream0)
        del primals_110
        buf184 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_113, buf184, 1, grid=grid(1), stream=stream0)
        del primals_113
        buf185 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_116, buf185, 1, grid=grid(1), stream=stream0)
        del primals_116
        buf186 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_119, buf186, 1, grid=grid(1), stream=stream0)
        del primals_119
        buf187 = empty_strided((), (), device="cuda", dtype=torch.int64)
        triton__17.run(primals_122, buf187, 1, grid=grid(1), stream=stream0)
        del primals_122
        return (
            buf7,
            buf8,
            buf168,
            buf17,
            buf18,
            buf169,
            buf25,
            buf26,
            buf170,
            buf33,
            buf34,
            buf171,
            buf41,
            buf42,
            buf172,
            buf49,
            buf50,
            buf173,
            buf57,
            buf58,
            buf174,
            buf64,
            buf65,
            buf175,
            buf73,
            buf74,
            buf176,
            buf81,
            buf82,
            buf177,
            buf89,
            buf90,
            buf178,
            buf97,
            buf98,
            buf179,
            buf104,
            buf105,
            buf180,
            buf113,
            buf114,
            buf181,
            buf121,
            buf122,
            buf182,
            buf129,
            buf130,
            buf183,
            buf137,
            buf138,
            buf184,
            buf144,
            buf145,
            buf185,
            buf153,
            buf154,
            buf186,
            buf161,
            buf162,
            buf187,
            buf166,
            primals_1,
            primals_2,
            primals_4,
            primals_5,
            primals_7,
            primals_8,
            primals_10,
            primals_11,
            primals_13,
            primals_14,
            primals_16,
            primals_17,
            primals_19,
            primals_20,
            primals_22,
            primals_23,
            primals_25,
            primals_26,
            primals_28,
            primals_29,
            primals_31,
            primals_32,
            primals_34,
            primals_35,
            primals_37,
            primals_38,
            primals_40,
            primals_41,
            primals_43,
            primals_44,
            primals_46,
            primals_47,
            primals_49,
            primals_50,
            primals_52,
            primals_53,
            primals_55,
            primals_56,
            primals_58,
            primals_59,
            primals_123,
            buf0,
            buf6,
            buf9,
            buf10,
            buf11,
            buf12,
            buf16,
            buf19,
            buf20,
            buf24,
            buf27,
            buf28,
            buf32,
            buf35,
            buf36,
            buf40,
            buf43,
            buf44,
            buf48,
            buf51,
            buf52,
            buf56,
            buf59,
            buf63,
            buf67,
            buf68,
            buf72,
            buf75,
            buf76,
            buf80,
            buf83,
            buf84,
            buf88,
            buf91,
            buf92,
            buf96,
            buf99,
            buf103,
            buf107,
            buf108,
            buf112,
            buf115,
            buf116,
            buf120,
            buf123,
            buf124,
            buf128,
            buf131,
            buf132,
            buf136,
            buf139,
            buf143,
            buf147,
            buf148,
            buf152,
            buf155,
            buf156,
            buf160,
            buf165,
            as_strided(primals_61, (1000, 512), (512, 1)),
            buf167,
            as_strided(buf158, (1, 512, 1, 1), (512, 1, 1, 1)),
            as_strided(buf150, (1, 512, 1, 1), (512, 1, 1, 1)),
            as_strided(buf141, (1, 512, 1, 1), (512, 1, 1, 1)),
            as_strided(buf134, (1, 512, 1, 1), (512, 1, 1, 1)),
            as_strided(buf126, (1, 512, 1, 1), (512, 1, 1, 1)),
            as_strided(buf118, (1, 256, 1, 1), (256, 1, 1, 1)),
            as_strided(buf110, (1, 256, 1, 1), (256, 1, 1, 1)),
            as_strided(buf101, (1, 256, 1, 1), (256, 1, 1, 1)),
            as_strided(buf94, (1, 256, 1, 1), (256, 1, 1, 1)),
            as_strided(buf86, (1, 256, 1, 1), (256, 1, 1, 1)),
            as_strided(buf78, (1, 128, 1, 1), (128, 1, 1, 1)),
            as_strided(buf70, (1, 128, 1, 1), (128, 1, 1, 1)),
            as_strided(buf61, (1, 128, 1, 1), (128, 1, 1, 1)),
            as_strided(buf54, (1, 128, 1, 1), (128, 1, 1, 1)),
            as_strided(buf46, (1, 128, 1, 1), (128, 1, 1, 1)),
            as_strided(buf38, (1, 64, 1, 1), (64, 1, 1, 1)),
            as_strided(buf30, (1, 64, 1, 1), (64, 1, 1, 1)),
            as_strided(buf22, (1, 64, 1, 1), (64, 1, 1, 1)),
            as_strided(buf14, (1, 64, 1, 1), (64, 1, 1, 1)),
            as_strided(buf3, (1, 64, 1, 1), (64, 1, 1, 1)),
        )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (64, 3, 7, 7), (147, 49, 7, 1), device="cuda:0", dtype=torch.float32
    )
    primals_2 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_3 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_4 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_5 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_6 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_7 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_8 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_9 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_10 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_11 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_12 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_13 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_14 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_15 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_16 = rand_strided(
        (128, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_17 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_18 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_19 = rand_strided(
        (128, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_20 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_21 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_22 = rand_strided(
        (128, 64, 1, 1), (64, 1, 1, 1), device="cuda:0", dtype=torch.float32
    )
    primals_23 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_24 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_25 = rand_strided(
        (128, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_26 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_27 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_28 = rand_strided(
        (128, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_29 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_30 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_31 = rand_strided(
        (256, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_32 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_33 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_34 = rand_strided(
        (256, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_35 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_36 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_37 = rand_strided(
        (256, 128, 1, 1), (128, 1, 1, 1), device="cuda:0", dtype=torch.float32
    )
    primals_38 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_39 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_40 = rand_strided(
        (256, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_41 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_42 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_43 = rand_strided(
        (256, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_44 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_45 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_46 = rand_strided(
        (512, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_47 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_48 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_49 = rand_strided(
        (512, 512, 3, 3), (4608, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_50 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_51 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_52 = rand_strided(
        (512, 256, 1, 1), (256, 1, 1, 1), device="cuda:0", dtype=torch.float32
    )
    primals_53 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_54 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_55 = rand_strided(
        (512, 512, 3, 3), (4608, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_56 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_57 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_58 = rand_strided(
        (512, 512, 3, 3), (4608, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    primals_59 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_60 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_61 = rand_strided(
        (1000, 512), (512, 1), device="cuda:0", dtype=torch.float32
    )
    primals_62 = rand_strided((1000,), (1,), device="cuda:0", dtype=torch.float32)
    primals_63 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_64 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_65 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_66 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_67 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_68 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_69 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_70 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_71 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_72 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_73 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_74 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_75 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_76 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    primals_77 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_78 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_79 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_80 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_81 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_82 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_83 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_84 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_85 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_86 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_87 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_88 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_89 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_90 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_91 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    primals_92 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_93 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_94 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_95 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_96 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_97 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_98 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_99 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_100 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_101 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_102 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_103 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_104 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_105 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_106 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    primals_107 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_108 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_109 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_110 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_111 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_112 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_113 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_114 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_115 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_116 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_117 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_118 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_119 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_120 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_121 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    primals_122 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    primals_123 = rand_strided(
        (1, 3, 224, 224), (150528, 50176, 224, 1), device="cuda:0", dtype=torch.float32
    )
    print_performance(
        lambda: call(
            [
                primals_1,
                primals_2,
                primals_3,
                primals_4,
                primals_5,
                primals_6,
                primals_7,
                primals_8,
                primals_9,
                primals_10,
                primals_11,
                primals_12,
                primals_13,
                primals_14,
                primals_15,
                primals_16,
                primals_17,
                primals_18,
                primals_19,
                primals_20,
                primals_21,
                primals_22,
                primals_23,
                primals_24,
                primals_25,
                primals_26,
                primals_27,
                primals_28,
                primals_29,
                primals_30,
                primals_31,
                primals_32,
                primals_33,
                primals_34,
                primals_35,
                primals_36,
                primals_37,
                primals_38,
                primals_39,
                primals_40,
                primals_41,
                primals_42,
                primals_43,
                primals_44,
                primals_45,
                primals_46,
                primals_47,
                primals_48,
                primals_49,
                primals_50,
                primals_51,
                primals_52,
                primals_53,
                primals_54,
                primals_55,
                primals_56,
                primals_57,
                primals_58,
                primals_59,
                primals_60,
                primals_61,
                primals_62,
                primals_63,
                primals_64,
                primals_65,
                primals_66,
                primals_67,
                primals_68,
                primals_69,
                primals_70,
                primals_71,
                primals_72,
                primals_73,
                primals_74,
                primals_75,
                primals_76,
                primals_77,
                primals_78,
                primals_79,
                primals_80,
                primals_81,
                primals_82,
                primals_83,
                primals_84,
                primals_85,
                primals_86,
                primals_87,
                primals_88,
                primals_89,
                primals_90,
                primals_91,
                primals_92,
                primals_93,
                primals_94,
                primals_95,
                primals_96,
                primals_97,
                primals_98,
                primals_99,
                primals_100,
                primals_101,
                primals_102,
                primals_103,
                primals_104,
                primals_105,
                primals_106,
                primals_107,
                primals_108,
                primals_109,
                primals_110,
                primals_111,
                primals_112,
                primals_113,
                primals_114,
                primals_115,
                primals_116,
                primals_117,
                primals_118,
                primals_119,
                primals_120,
                primals_121,
                primals_122,
                primals_123,
            ]
        )
    )
