
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
        self.load_state_dict(torch.load(r'resnet18-prims/state_dict.pt'))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1):
        convolution = torch.ops.aten.convolution.default(arg122_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg122_1 = arg0_1 = None
        _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution, arg1_1, arg2_1, arg62_1, arg63_1, 0.1, 1e-05);  convolution = arg1_1 = arg2_1 = arg62_1 = arg63_1 = None
        getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
        relu = torch.ops.aten.relu.default(getitem);  getitem = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
        getitem_3 = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
        convolution_1 = torch.ops.aten.convolution.default(getitem_3, arg3_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = None
        _native_batch_norm_legit_no_training_1 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_1, arg4_1, arg5_1, arg65_1, arg66_1, 0.1, 1e-05);  convolution_1 = arg4_1 = arg5_1 = arg65_1 = arg66_1 = None
        getitem_5 = _native_batch_norm_legit_no_training_1[0];  _native_batch_norm_legit_no_training_1 = None
        relu_1 = torch.ops.aten.relu.default(getitem_5);  getitem_5 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = None
        _native_batch_norm_legit_no_training_2 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_2, arg7_1, arg8_1, arg68_1, arg69_1, 0.1, 1e-05);  convolution_2 = arg7_1 = arg8_1 = arg68_1 = arg69_1 = None
        getitem_8 = _native_batch_norm_legit_no_training_2[0];  _native_batch_norm_legit_no_training_2 = None
        add = torch.ops.aten.add.Tensor(getitem_8, getitem_3);  getitem_8 = getitem_3 = None
        relu_2 = torch.ops.aten.relu.default(add);  add = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg9_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg9_1 = None
        _native_batch_norm_legit_no_training_3 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_3, arg10_1, arg11_1, arg71_1, arg72_1, 0.1, 1e-05);  convolution_3 = arg10_1 = arg11_1 = arg71_1 = arg72_1 = None
        getitem_11 = _native_batch_norm_legit_no_training_3[0];  _native_batch_norm_legit_no_training_3 = None
        relu_3 = torch.ops.aten.relu.default(getitem_11);  getitem_11 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg12_1 = None
        _native_batch_norm_legit_no_training_4 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_4, arg13_1, arg14_1, arg74_1, arg75_1, 0.1, 1e-05);  convolution_4 = arg13_1 = arg14_1 = arg74_1 = arg75_1 = None
        getitem_14 = _native_batch_norm_legit_no_training_4[0];  _native_batch_norm_legit_no_training_4 = None
        add_1 = torch.ops.aten.add.Tensor(getitem_14, relu_2);  getitem_14 = relu_2 = None
        relu_4 = torch.ops.aten.relu.default(add_1);  add_1 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, arg15_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg15_1 = None
        _native_batch_norm_legit_no_training_5 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_5, arg16_1, arg17_1, arg77_1, arg78_1, 0.1, 1e-05);  convolution_5 = arg16_1 = arg17_1 = arg77_1 = arg78_1 = None
        getitem_17 = _native_batch_norm_legit_no_training_5[0];  _native_batch_norm_legit_no_training_5 = None
        relu_5 = torch.ops.aten.relu.default(getitem_17);  getitem_17 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg18_1 = None
        _native_batch_norm_legit_no_training_6 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_6, arg19_1, arg20_1, arg80_1, arg81_1, 0.1, 1e-05);  convolution_6 = arg19_1 = arg20_1 = arg80_1 = arg81_1 = None
        getitem_20 = _native_batch_norm_legit_no_training_6[0];  _native_batch_norm_legit_no_training_6 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_4, arg21_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg21_1 = None
        _native_batch_norm_legit_no_training_7 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_7, arg22_1, arg23_1, arg83_1, arg84_1, 0.1, 1e-05);  convolution_7 = arg22_1 = arg23_1 = arg83_1 = arg84_1 = None
        getitem_23 = _native_batch_norm_legit_no_training_7[0];  _native_batch_norm_legit_no_training_7 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_20, getitem_23);  getitem_20 = getitem_23 = None
        relu_6 = torch.ops.aten.relu.default(add_2);  add_2 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_6, arg24_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg24_1 = None
        _native_batch_norm_legit_no_training_8 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_8, arg25_1, arg26_1, arg86_1, arg87_1, 0.1, 1e-05);  convolution_8 = arg25_1 = arg26_1 = arg86_1 = arg87_1 = None
        getitem_26 = _native_batch_norm_legit_no_training_8[0];  _native_batch_norm_legit_no_training_8 = None
        relu_7 = torch.ops.aten.relu.default(getitem_26);  getitem_26 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, arg27_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg27_1 = None
        _native_batch_norm_legit_no_training_9 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_9, arg28_1, arg29_1, arg89_1, arg90_1, 0.1, 1e-05);  convolution_9 = arg28_1 = arg29_1 = arg89_1 = arg90_1 = None
        getitem_29 = _native_batch_norm_legit_no_training_9[0];  _native_batch_norm_legit_no_training_9 = None
        add_3 = torch.ops.aten.add.Tensor(getitem_29, relu_6);  getitem_29 = relu_6 = None
        relu_8 = torch.ops.aten.relu.default(add_3);  add_3 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_8, arg30_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg30_1 = None
        _native_batch_norm_legit_no_training_10 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_10, arg31_1, arg32_1, arg92_1, arg93_1, 0.1, 1e-05);  convolution_10 = arg31_1 = arg32_1 = arg92_1 = arg93_1 = None
        getitem_32 = _native_batch_norm_legit_no_training_10[0];  _native_batch_norm_legit_no_training_10 = None
        relu_9 = torch.ops.aten.relu.default(getitem_32);  getitem_32 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_9, arg33_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_9 = arg33_1 = None
        _native_batch_norm_legit_no_training_11 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_11, arg34_1, arg35_1, arg95_1, arg96_1, 0.1, 1e-05);  convolution_11 = arg34_1 = arg35_1 = arg95_1 = arg96_1 = None
        getitem_35 = _native_batch_norm_legit_no_training_11[0];  _native_batch_norm_legit_no_training_11 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_8, arg36_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg36_1 = None
        _native_batch_norm_legit_no_training_12 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_12, arg37_1, arg38_1, arg98_1, arg99_1, 0.1, 1e-05);  convolution_12 = arg37_1 = arg38_1 = arg98_1 = arg99_1 = None
        getitem_38 = _native_batch_norm_legit_no_training_12[0];  _native_batch_norm_legit_no_training_12 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_35, getitem_38);  getitem_35 = getitem_38 = None
        relu_10 = torch.ops.aten.relu.default(add_4);  add_4 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_10, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg39_1 = None
        _native_batch_norm_legit_no_training_13 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_13, arg40_1, arg41_1, arg101_1, arg102_1, 0.1, 1e-05);  convolution_13 = arg40_1 = arg41_1 = arg101_1 = arg102_1 = None
        getitem_41 = _native_batch_norm_legit_no_training_13[0];  _native_batch_norm_legit_no_training_13 = None
        relu_11 = torch.ops.aten.relu.default(getitem_41);  getitem_41 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_11, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg42_1 = None
        _native_batch_norm_legit_no_training_14 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_14, arg43_1, arg44_1, arg104_1, arg105_1, 0.1, 1e-05);  convolution_14 = arg43_1 = arg44_1 = arg104_1 = arg105_1 = None
        getitem_44 = _native_batch_norm_legit_no_training_14[0];  _native_batch_norm_legit_no_training_14 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_44, relu_10);  getitem_44 = relu_10 = None
        relu_12 = torch.ops.aten.relu.default(add_5);  add_5 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_12, arg45_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg45_1 = None
        _native_batch_norm_legit_no_training_15 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_15, arg46_1, arg47_1, arg107_1, arg108_1, 0.1, 1e-05);  convolution_15 = arg46_1 = arg47_1 = arg107_1 = arg108_1 = None
        getitem_47 = _native_batch_norm_legit_no_training_15[0];  _native_batch_norm_legit_no_training_15 = None
        relu_13 = torch.ops.aten.relu.default(getitem_47);  getitem_47 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_13, arg48_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg48_1 = None
        _native_batch_norm_legit_no_training_16 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_16, arg49_1, arg50_1, arg110_1, arg111_1, 0.1, 1e-05);  convolution_16 = arg49_1 = arg50_1 = arg110_1 = arg111_1 = None
        getitem_50 = _native_batch_norm_legit_no_training_16[0];  _native_batch_norm_legit_no_training_16 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_12, arg51_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg51_1 = None
        _native_batch_norm_legit_no_training_17 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_17, arg52_1, arg53_1, arg113_1, arg114_1, 0.1, 1e-05);  convolution_17 = arg52_1 = arg53_1 = arg113_1 = arg114_1 = None
        getitem_53 = _native_batch_norm_legit_no_training_17[0];  _native_batch_norm_legit_no_training_17 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_50, getitem_53);  getitem_50 = getitem_53 = None
        relu_14 = torch.ops.aten.relu.default(add_6);  add_6 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_14, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg54_1 = None
        _native_batch_norm_legit_no_training_18 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_18, arg55_1, arg56_1, arg116_1, arg117_1, 0.1, 1e-05);  convolution_18 = arg55_1 = arg56_1 = arg116_1 = arg117_1 = None
        getitem_56 = _native_batch_norm_legit_no_training_18[0];  _native_batch_norm_legit_no_training_18 = None
        relu_15 = torch.ops.aten.relu.default(getitem_56);  getitem_56 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_15, arg57_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_15 = arg57_1 = None
        _native_batch_norm_legit_no_training_19 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_19, arg58_1, arg59_1, arg119_1, arg120_1, 0.1, 1e-05);  convolution_19 = arg58_1 = arg59_1 = arg119_1 = arg120_1 = None
        getitem_59 = _native_batch_norm_legit_no_training_19[0];  _native_batch_norm_legit_no_training_19 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_59, relu_14);  getitem_59 = relu_14 = None
        relu_16 = torch.ops.aten.relu.default(add_7);  add_7 = None
        mean = torch.ops.aten.mean.dim(relu_16, [-1, -2], True);  relu_16 = None
        view = torch.ops.aten.view.default(mean, [1, 512]);  mean = None
        t = torch.ops.aten.t.default(arg60_1);  arg60_1 = None
        addmm = torch.ops.aten.addmm.default(arg61_1, view, t);  arg61_1 = view = t = None
        return (addmm,)
        
