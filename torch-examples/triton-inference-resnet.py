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
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
"""
)


triton__1 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
"""
)


triton__2 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
"""
)


triton__3 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp15 = tl.load(in_ptr4 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
"""
)


triton__6 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp15 = tl.load(in_ptr4 + (x2), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp18 = tl.load(in_ptr6 + (x1), xmask)
    tmp24 = tl.load(in_ptr7 + (x1), xmask)
    tmp26 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.where(0 != 0, 0, tl.where(0 > tmp28, 0, tmp28))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp29, xmask)
"""
)


triton__7 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
"""
)


triton__8 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
"""
)


triton__9 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp15 = tl.load(in_ptr4 + (x2), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp18 = tl.load(in_ptr6 + (x1), xmask)
    tmp24 = tl.load(in_ptr7 + (x1), xmask)
    tmp26 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.where(0 != 0, 0, tl.where(0 > tmp28, 0, tmp28))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp29, xmask)
"""
)


triton__10 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
"""
)


triton__11 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
"""
)


triton__12 = async_compile.triton(
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp15 = tl.load(in_ptr4 + (x2), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp18 = tl.load(in_ptr6 + (x1), xmask)
    tmp24 = tl.load(in_ptr7 + (x1), xmask)
    tmp26 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.where(0 != 0, 0, tl.where(0 > tmp28, 0, tmp28))
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp29, xmask)
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
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask)
    tmp15 = tl.load(in_ptr5 + (r1 + (49*x0)), rmask & xmask, other=0)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp22, xmask)
"""
)


async_compile.wait(globals())
del async_compile


def call(args):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        arg29_1,
        arg30_1,
        arg31_1,
        arg32_1,
        arg33_1,
        arg34_1,
        arg35_1,
        arg36_1,
        arg37_1,
        arg38_1,
        arg39_1,
        arg40_1,
        arg41_1,
        arg42_1,
        arg43_1,
        arg44_1,
        arg45_1,
        arg46_1,
        arg47_1,
        arg48_1,
        arg49_1,
        arg50_1,
        arg51_1,
        arg52_1,
        arg53_1,
        arg54_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        arg59_1,
        arg60_1,
        arg61_1,
        arg62_1,
        arg63_1,
        arg64_1,
        arg65_1,
        arg66_1,
        arg67_1,
        arg68_1,
        arg69_1,
        arg70_1,
        arg71_1,
        arg72_1,
        arg73_1,
        arg74_1,
        arg75_1,
        arg76_1,
        arg77_1,
        arg78_1,
        arg79_1,
        arg80_1,
        arg81_1,
        arg82_1,
        arg83_1,
        arg84_1,
        arg85_1,
        arg86_1,
        arg87_1,
        arg88_1,
        arg89_1,
        arg90_1,
        arg91_1,
        arg92_1,
        arg93_1,
        arg94_1,
        arg95_1,
        arg96_1,
        arg97_1,
        arg98_1,
        arg99_1,
        arg100_1,
        arg101_1,
        arg102_1,
        arg103_1,
        arg104_1,
        arg105_1,
        arg106_1,
        arg107_1,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
        arg112_1,
        arg113_1,
        arg114_1,
        arg115_1,
        arg116_1,
        arg117_1,
        arg118_1,
        arg119_1,
        arg120_1,
        arg121_1,
        arg122_1,
    ) = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = aten.convolution(
            arg122_1, arg0_1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf0, (1, 64, 112, 112), (802816, 12544, 112, 1))
        del arg0_1
        del arg122_1
        buf1 = buf0
        del buf0  # reuse
        stream0 = get_cuda_stream(0)
        triton__0.run(
            buf1,
            arg62_1,
            arg63_1,
            arg1_1,
            arg2_1,
            802816,
            grid=grid(802816),
            stream=stream0,
        )
        del arg1_1
        del arg2_1
        del arg62_1
        del arg63_1
        buf2 = empty_strided(
            (1, 64, 56, 56), (200704, 3136, 56, 1), device="cuda", dtype=torch.float32
        )
        triton__1.run(buf1, buf2, 200704, grid=grid(200704), stream=stream0)
        del buf1
        buf4 = aten.convolution(
            buf2, arg3_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf4, (1, 64, 56, 56), (200704, 3136, 56, 1))
        del arg3_1
        buf5 = buf4
        del buf4  # reuse
        triton__2.run(
            buf5,
            arg65_1,
            arg66_1,
            arg4_1,
            arg5_1,
            200704,
            grid=grid(200704),
            stream=stream0,
        )
        del arg4_1
        del arg5_1
        del arg65_1
        del arg66_1
        buf6 = aten.convolution(
            buf5, arg6_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf6, (1, 64, 56, 56), (200704, 3136, 56, 1))
        del arg6_1
        del buf5
        buf7 = buf2
        del buf2  # reuse
        triton__3.run(
            buf7,
            buf6,
            arg68_1,
            arg69_1,
            arg7_1,
            arg8_1,
            200704,
            grid=grid(200704),
            stream=stream0,
        )
        del arg68_1
        del arg69_1
        del arg7_1
        del arg8_1
        del buf6
        buf8 = aten.convolution(
            buf7, arg9_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf8, (1, 64, 56, 56), (200704, 3136, 56, 1))
        del arg9_1
        buf9 = buf8
        del buf8  # reuse
        triton__2.run(
            buf9,
            arg71_1,
            arg72_1,
            arg10_1,
            arg11_1,
            200704,
            grid=grid(200704),
            stream=stream0,
        )
        del arg10_1
        del arg11_1
        del arg71_1
        del arg72_1
        buf10 = aten.convolution(
            buf9, arg12_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf10, (1, 64, 56, 56), (200704, 3136, 56, 1))
        del arg12_1
        del buf9
        buf11 = buf10
        del buf10  # reuse
        triton__4.run(
            buf11,
            arg74_1,
            arg75_1,
            arg13_1,
            arg14_1,
            buf7,
            200704,
            grid=grid(200704),
            stream=stream0,
        )
        del arg13_1
        del arg14_1
        del arg74_1
        del arg75_1
        del buf7
        buf12 = aten.convolution(
            buf11, arg15_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf12, (1, 128, 28, 28), (100352, 784, 28, 1))
        del arg15_1
        buf13 = buf12
        del buf12  # reuse
        triton__5.run(
            buf13,
            arg77_1,
            arg78_1,
            arg16_1,
            arg17_1,
            100352,
            grid=grid(100352),
            stream=stream0,
        )
        del arg16_1
        del arg17_1
        del arg77_1
        del arg78_1
        buf14 = aten.convolution(
            buf13, arg18_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf14, (1, 128, 28, 28), (100352, 784, 28, 1))
        del arg18_1
        del buf13
        buf15 = aten.convolution(
            buf11, arg21_1, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf15, (1, 128, 28, 28), (100352, 784, 28, 1))
        del arg21_1
        del buf11
        buf16 = buf14
        del buf14  # reuse
        buf17 = buf16
        del buf16  # reuse
        triton__6.run(
            buf17,
            arg80_1,
            arg81_1,
            arg19_1,
            arg20_1,
            buf15,
            arg83_1,
            arg84_1,
            arg22_1,
            arg23_1,
            100352,
            grid=grid(100352),
            stream=stream0,
        )
        del arg19_1
        del arg20_1
        del arg22_1
        del arg23_1
        del arg80_1
        del arg81_1
        del arg83_1
        del arg84_1
        del buf15
        buf18 = aten.convolution(
            buf17, arg24_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf18, (1, 128, 28, 28), (100352, 784, 28, 1))
        del arg24_1
        buf19 = buf18
        del buf18  # reuse
        triton__5.run(
            buf19,
            arg86_1,
            arg87_1,
            arg25_1,
            arg26_1,
            100352,
            grid=grid(100352),
            stream=stream0,
        )
        del arg25_1
        del arg26_1
        del arg86_1
        del arg87_1
        buf20 = aten.convolution(
            buf19, arg27_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf20, (1, 128, 28, 28), (100352, 784, 28, 1))
        del arg27_1
        del buf19
        buf21 = buf17
        del buf17  # reuse
        triton__7.run(
            buf21,
            buf20,
            arg89_1,
            arg90_1,
            arg28_1,
            arg29_1,
            100352,
            grid=grid(100352),
            stream=stream0,
        )
        del arg28_1
        del arg29_1
        del arg89_1
        del arg90_1
        del buf20
        buf22 = aten.convolution(
            buf21, arg30_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf22, (1, 256, 14, 14), (50176, 196, 14, 1))
        del arg30_1
        buf23 = buf22
        del buf22  # reuse
        triton__8.run(
            buf23,
            arg92_1,
            arg93_1,
            arg31_1,
            arg32_1,
            50176,
            grid=grid(50176),
            stream=stream0,
        )
        del arg31_1
        del arg32_1
        del arg92_1
        del arg93_1
        buf24 = aten.convolution(
            buf23, arg33_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf24, (1, 256, 14, 14), (50176, 196, 14, 1))
        del arg33_1
        del buf23
        buf25 = aten.convolution(
            buf21, arg36_1, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf25, (1, 256, 14, 14), (50176, 196, 14, 1))
        del arg36_1
        del buf21
        buf26 = buf24
        del buf24  # reuse
        buf27 = buf26
        del buf26  # reuse
        triton__9.run(
            buf27,
            arg95_1,
            arg96_1,
            arg34_1,
            arg35_1,
            buf25,
            arg98_1,
            arg99_1,
            arg37_1,
            arg38_1,
            50176,
            grid=grid(50176),
            stream=stream0,
        )
        del arg34_1
        del arg35_1
        del arg37_1
        del arg38_1
        del arg95_1
        del arg96_1
        del arg98_1
        del arg99_1
        del buf25
        buf28 = aten.convolution(
            buf27, arg39_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf28, (1, 256, 14, 14), (50176, 196, 14, 1))
        del arg39_1
        buf29 = buf28
        del buf28  # reuse
        triton__8.run(
            buf29,
            arg101_1,
            arg102_1,
            arg40_1,
            arg41_1,
            50176,
            grid=grid(50176),
            stream=stream0,
        )
        del arg101_1
        del arg102_1
        del arg40_1
        del arg41_1
        buf30 = aten.convolution(
            buf29, arg42_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf30, (1, 256, 14, 14), (50176, 196, 14, 1))
        del arg42_1
        del buf29
        buf31 = buf27
        del buf27  # reuse
        triton__10.run(
            buf31,
            buf30,
            arg104_1,
            arg105_1,
            arg43_1,
            arg44_1,
            50176,
            grid=grid(50176),
            stream=stream0,
        )
        del arg104_1
        del arg105_1
        del arg43_1
        del arg44_1
        del buf30
        buf32 = aten.convolution(
            buf31, arg45_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf32, (1, 512, 7, 7), (25088, 49, 7, 1))
        del arg45_1
        buf33 = buf32
        del buf32  # reuse
        triton__11.run(
            buf33,
            arg107_1,
            arg108_1,
            arg46_1,
            arg47_1,
            25088,
            grid=grid(25088),
            stream=stream0,
        )
        del arg107_1
        del arg108_1
        del arg46_1
        del arg47_1
        buf34 = aten.convolution(
            buf33, arg48_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf34, (1, 512, 7, 7), (25088, 49, 7, 1))
        del arg48_1
        del buf33
        buf35 = aten.convolution(
            buf31, arg51_1, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf35, (1, 512, 7, 7), (25088, 49, 7, 1))
        del arg51_1
        del buf31
        buf36 = buf34
        del buf34  # reuse
        buf37 = buf36
        del buf36  # reuse
        triton__12.run(
            buf37,
            arg110_1,
            arg111_1,
            arg49_1,
            arg50_1,
            buf35,
            arg113_1,
            arg114_1,
            arg52_1,
            arg53_1,
            25088,
            grid=grid(25088),
            stream=stream0,
        )
        del arg110_1
        del arg111_1
        del arg113_1
        del arg114_1
        del arg49_1
        del arg50_1
        del arg52_1
        del arg53_1
        del buf35
        buf38 = aten.convolution(
            buf37, arg54_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf38, (1, 512, 7, 7), (25088, 49, 7, 1))
        del arg54_1
        buf39 = buf38
        del buf38  # reuse
        triton__11.run(
            buf39,
            arg116_1,
            arg117_1,
            arg55_1,
            arg56_1,
            25088,
            grid=grid(25088),
            stream=stream0,
        )
        del arg116_1
        del arg117_1
        del arg55_1
        del arg56_1
        buf40 = aten.convolution(
            buf39, arg57_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
        )
        assert_size_stride(buf40, (1, 512, 7, 7), (25088, 49, 7, 1))
        del arg57_1
        del buf39
        buf41 = empty_strided(
            (1, 512, 1, 1), (512, 1, 512, 512), device="cuda", dtype=torch.float32
        )
        buf42 = as_strided(buf41, (1, 512, 1, 1), (512, 1, 1, 1))
        del buf41  # reuse
        triton__13.run(
            buf42,
            buf40,
            arg119_1,
            arg120_1,
            arg58_1,
            arg59_1,
            buf37,
            512,
            49,
            grid=grid(512),
            stream=stream0,
        )
        del arg119_1
        del arg120_1
        del arg58_1
        del arg59_1
        del buf37
        del buf40
        buf43 = empty_strided((1, 1000), (1000, 1), device="cuda", dtype=torch.float32)
        extern_kernels.addmm(
            arg61_1,
            as_strided(buf42, (1, 512), (0, 1)),
            as_strided(arg60_1, (512, 1000), (1, 512)),
            alpha=1,
            beta=1,
            out=buf43,
        )
        del arg60_1
        del arg61_1
        return (buf43,)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    from fijit_py import Fijit

    f = Fijit(True, False)
    f.run()

    arg0_1 = rand_strided(
        (64, 3, 7, 7), (147, 49, 7, 1), device="cuda:0", dtype=torch.float32
    )
    arg1_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg2_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg3_1 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg4_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg5_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg6_1 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg7_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg8_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg9_1 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg10_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg11_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg12_1 = rand_strided(
        (64, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg13_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg14_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg15_1 = rand_strided(
        (128, 64, 3, 3), (576, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg16_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg17_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg18_1 = rand_strided(
        (128, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg19_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg20_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg21_1 = rand_strided(
        (128, 64, 1, 1), (64, 1, 1, 1), device="cuda:0", dtype=torch.float32
    )
    arg22_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg23_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg24_1 = rand_strided(
        (128, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg25_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg26_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg27_1 = rand_strided(
        (128, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg28_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg29_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg30_1 = rand_strided(
        (256, 128, 3, 3), (1152, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg31_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg32_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg33_1 = rand_strided(
        (256, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg34_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg35_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg36_1 = rand_strided(
        (256, 128, 1, 1), (128, 1, 1, 1), device="cuda:0", dtype=torch.float32
    )
    arg37_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg38_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg39_1 = rand_strided(
        (256, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg40_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg41_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg42_1 = rand_strided(
        (256, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg43_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg44_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg45_1 = rand_strided(
        (512, 256, 3, 3), (2304, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg46_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg47_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg48_1 = rand_strided(
        (512, 512, 3, 3), (4608, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg49_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg50_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg51_1 = rand_strided(
        (512, 256, 1, 1), (256, 1, 1, 1), device="cuda:0", dtype=torch.float32
    )
    arg52_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg53_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg54_1 = rand_strided(
        (512, 512, 3, 3), (4608, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg55_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg56_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg57_1 = rand_strided(
        (512, 512, 3, 3), (4608, 9, 3, 1), device="cuda:0", dtype=torch.float32
    )
    arg58_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg59_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg60_1 = rand_strided((1000, 512), (512, 1), device="cuda:0", dtype=torch.float32)
    arg61_1 = rand_strided((1000,), (1,), device="cuda:0", dtype=torch.float32)
    arg62_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg63_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg64_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg65_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg66_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg67_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg68_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg69_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg70_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg71_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg72_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg73_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg74_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg75_1 = rand_strided((64,), (1,), device="cuda:0", dtype=torch.float32)
    arg76_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg77_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg78_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg79_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg80_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg81_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg82_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg83_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg84_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg85_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg86_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg87_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg88_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg89_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg90_1 = rand_strided((128,), (1,), device="cuda:0", dtype=torch.float32)
    arg91_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg92_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg93_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg94_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg95_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg96_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg97_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg98_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg99_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg100_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg101_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg102_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg103_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg104_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg105_1 = rand_strided((256,), (1,), device="cuda:0", dtype=torch.float32)
    arg106_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg107_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg108_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg109_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg110_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg111_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg112_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg113_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg114_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg115_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg116_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg117_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg118_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg119_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg120_1 = rand_strided((512,), (1,), device="cuda:0", dtype=torch.float32)
    arg121_1 = rand_strided((), (), device="cuda:0", dtype=torch.int64)
    arg122_1 = rand_strided(
        (1, 3, 224, 224), (150528, 50176, 224, 1), device="cuda:0", dtype=torch.float32
    )
    print_performance(
        lambda: call(
            [
                arg0_1,
                arg1_1,
                arg2_1,
                arg3_1,
                arg4_1,
                arg5_1,
                arg6_1,
                arg7_1,
                arg8_1,
                arg9_1,
                arg10_1,
                arg11_1,
                arg12_1,
                arg13_1,
                arg14_1,
                arg15_1,
                arg16_1,
                arg17_1,
                arg18_1,
                arg19_1,
                arg20_1,
                arg21_1,
                arg22_1,
                arg23_1,
                arg24_1,
                arg25_1,
                arg26_1,
                arg27_1,
                arg28_1,
                arg29_1,
                arg30_1,
                arg31_1,
                arg32_1,
                arg33_1,
                arg34_1,
                arg35_1,
                arg36_1,
                arg37_1,
                arg38_1,
                arg39_1,
                arg40_1,
                arg41_1,
                arg42_1,
                arg43_1,
                arg44_1,
                arg45_1,
                arg46_1,
                arg47_1,
                arg48_1,
                arg49_1,
                arg50_1,
                arg51_1,
                arg52_1,
                arg53_1,
                arg54_1,
                arg55_1,
                arg56_1,
                arg57_1,
                arg58_1,
                arg59_1,
                arg60_1,
                arg61_1,
                arg62_1,
                arg63_1,
                arg64_1,
                arg65_1,
                arg66_1,
                arg67_1,
                arg68_1,
                arg69_1,
                arg70_1,
                arg71_1,
                arg72_1,
                arg73_1,
                arg74_1,
                arg75_1,
                arg76_1,
                arg77_1,
                arg78_1,
                arg79_1,
                arg80_1,
                arg81_1,
                arg82_1,
                arg83_1,
                arg84_1,
                arg85_1,
                arg86_1,
                arg87_1,
                arg88_1,
                arg89_1,
                arg90_1,
                arg91_1,
                arg92_1,
                arg93_1,
                arg94_1,
                arg95_1,
                arg96_1,
                arg97_1,
                arg98_1,
                arg99_1,
                arg100_1,
                arg101_1,
                arg102_1,
                arg103_1,
                arg104_1,
                arg105_1,
                arg106_1,
                arg107_1,
                arg108_1,
                arg109_1,
                arg110_1,
                arg111_1,
                arg112_1,
                arg113_1,
                arg114_1,
                arg115_1,
                arg116_1,
                arg117_1,
                arg118_1,
                arg119_1,
                arg120_1,
                arg121_1,
                arg122_1,
            ]
        )
    )

    print(f.get_kernel_records())
