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
        self.model_conv1 = Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model_bn1 = BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_relu = torch.load(r"resnet18/model_relu.pt")  # ReLU(inplace=True)
        self.model_maxpool = torch.load(
            r"resnet18/model_maxpool.pt"
        )  # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.model_layer1_0_conv1 = Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer1_0_bn1 = BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer1_0_relu = torch.load(
            r"resnet18/model_layer1_0_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer1_0_conv2 = Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer1_0_bn2 = BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer1_1_conv1 = Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer1_1_bn1 = BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer1_1_relu = torch.load(
            r"resnet18/model_layer1_1_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer1_1_conv2 = Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer1_1_bn2 = BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer2_0_conv1 = Conv2d(
            64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.model_layer2_0_bn1 = BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer2_0_relu = torch.load(
            r"resnet18/model_layer2_0_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer2_0_conv2 = Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer2_0_bn2 = BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer2_0_downsample_0 = Conv2d(
            64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
        )
        self.model_layer2_0_downsample_1 = BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer2_1_conv1 = Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer2_1_bn1 = BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer2_1_relu = torch.load(
            r"resnet18/model_layer2_1_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer2_1_conv2 = Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer2_1_bn2 = BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer3_0_conv1 = Conv2d(
            128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.model_layer3_0_bn1 = BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer3_0_relu = torch.load(
            r"resnet18/model_layer3_0_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer3_0_conv2 = Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer3_0_bn2 = BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer3_0_downsample_0 = Conv2d(
            128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
        )
        self.model_layer3_0_downsample_1 = BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer3_1_conv1 = Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer3_1_bn1 = BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer3_1_relu = torch.load(
            r"resnet18/model_layer3_1_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer3_1_conv2 = Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer3_1_bn2 = BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer4_0_conv1 = Conv2d(
            256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.model_layer4_0_bn1 = BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer4_0_relu = torch.load(
            r"resnet18/model_layer4_0_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer4_0_conv2 = Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer4_0_bn2 = BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer4_0_downsample_0 = Conv2d(
            256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
        )
        self.model_layer4_0_downsample_1 = BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer4_1_conv1 = Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer4_1_bn1 = BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_layer4_1_relu = torch.load(
            r"resnet18/model_layer4_1_relu.pt"
        )  # ReLU(inplace=True)
        self.model_layer4_1_conv2 = Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model_layer4_1_bn2 = BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.model_avgpool = torch.load(
            r"resnet18/model_avgpool.pt"
        )  # AdaptiveAvgPool2d(output_size=(1, 1))
        self.model_fc = Linear(in_features=512, out_features=1000, bias=True)
        self.load_state_dict(torch.load(r"resnet18/state_dict.pt"))

    def forward(self, inp: torch.Tensor):
        model_conv1 = self.model_conv1(inp)
        inp = None
        model_bn1 = self.model_bn1(model_conv1)
        model_conv1 = None
        model_relu = self.model_relu(model_bn1)
        model_bn1 = None
        model_maxpool = self.model_maxpool(model_relu)
        model_relu = None
        model_layer1_0_conv1 = self.model_layer1_0_conv1(model_maxpool)
        model_layer1_0_bn1 = self.model_layer1_0_bn1(model_layer1_0_conv1)
        model_layer1_0_conv1 = None
        model_layer1_0_relu = self.model_layer1_0_relu(model_layer1_0_bn1)
        model_layer1_0_bn1 = None
        model_layer1_0_conv2 = self.model_layer1_0_conv2(model_layer1_0_relu)
        model_layer1_0_relu = None
        model_layer1_0_bn2 = self.model_layer1_0_bn2(model_layer1_0_conv2)
        model_layer1_0_conv2 = None
        model_layer1_0_bn2 += model_maxpool
        iadd = model_layer1_0_bn2
        model_layer1_0_bn2 = model_maxpool = None
        model_layer1_0_relu_1 = self.model_layer1_0_relu(iadd)
        iadd = None
        model_layer1_1_conv1 = self.model_layer1_1_conv1(model_layer1_0_relu_1)
        model_layer1_1_bn1 = self.model_layer1_1_bn1(model_layer1_1_conv1)
        model_layer1_1_conv1 = None
        model_layer1_1_relu = self.model_layer1_1_relu(model_layer1_1_bn1)
        model_layer1_1_bn1 = None
        model_layer1_1_conv2 = self.model_layer1_1_conv2(model_layer1_1_relu)
        model_layer1_1_relu = None
        model_layer1_1_bn2 = self.model_layer1_1_bn2(model_layer1_1_conv2)
        model_layer1_1_conv2 = None
        model_layer1_1_bn2 += model_layer1_0_relu_1
        iadd_1 = model_layer1_1_bn2
        model_layer1_1_bn2 = model_layer1_0_relu_1 = None
        model_layer1_1_relu_1 = self.model_layer1_1_relu(iadd_1)
        iadd_1 = None
        model_layer2_0_conv1 = self.model_layer2_0_conv1(model_layer1_1_relu_1)
        model_layer2_0_bn1 = self.model_layer2_0_bn1(model_layer2_0_conv1)
        model_layer2_0_conv1 = None
        model_layer2_0_relu = self.model_layer2_0_relu(model_layer2_0_bn1)
        model_layer2_0_bn1 = None
        model_layer2_0_conv2 = self.model_layer2_0_conv2(model_layer2_0_relu)
        model_layer2_0_relu = None
        model_layer2_0_bn2 = self.model_layer2_0_bn2(model_layer2_0_conv2)
        model_layer2_0_conv2 = None
        model_layer2_0_downsample_0 = self.model_layer2_0_downsample_0(
            model_layer1_1_relu_1
        )
        model_layer1_1_relu_1 = None
        model_layer2_0_downsample_1 = self.model_layer2_0_downsample_1(
            model_layer2_0_downsample_0
        )
        model_layer2_0_downsample_0 = None
        model_layer2_0_bn2 += model_layer2_0_downsample_1
        iadd_2 = model_layer2_0_bn2
        model_layer2_0_bn2 = model_layer2_0_downsample_1 = None
        model_layer2_0_relu_1 = self.model_layer2_0_relu(iadd_2)
        iadd_2 = None
        model_layer2_1_conv1 = self.model_layer2_1_conv1(model_layer2_0_relu_1)
        model_layer2_1_bn1 = self.model_layer2_1_bn1(model_layer2_1_conv1)
        model_layer2_1_conv1 = None
        model_layer2_1_relu = self.model_layer2_1_relu(model_layer2_1_bn1)
        model_layer2_1_bn1 = None
        model_layer2_1_conv2 = self.model_layer2_1_conv2(model_layer2_1_relu)
        model_layer2_1_relu = None
        model_layer2_1_bn2 = self.model_layer2_1_bn2(model_layer2_1_conv2)
        model_layer2_1_conv2 = None
        model_layer2_1_bn2 += model_layer2_0_relu_1
        iadd_3 = model_layer2_1_bn2
        model_layer2_1_bn2 = model_layer2_0_relu_1 = None
        model_layer2_1_relu_1 = self.model_layer2_1_relu(iadd_3)
        iadd_3 = None
        model_layer3_0_conv1 = self.model_layer3_0_conv1(model_layer2_1_relu_1)
        model_layer3_0_bn1 = self.model_layer3_0_bn1(model_layer3_0_conv1)
        model_layer3_0_conv1 = None
        model_layer3_0_relu = self.model_layer3_0_relu(model_layer3_0_bn1)
        model_layer3_0_bn1 = None
        model_layer3_0_conv2 = self.model_layer3_0_conv2(model_layer3_0_relu)
        model_layer3_0_relu = None
        model_layer3_0_bn2 = self.model_layer3_0_bn2(model_layer3_0_conv2)
        model_layer3_0_conv2 = None
        model_layer3_0_downsample_0 = self.model_layer3_0_downsample_0(
            model_layer2_1_relu_1
        )
        model_layer2_1_relu_1 = None
        model_layer3_0_downsample_1 = self.model_layer3_0_downsample_1(
            model_layer3_0_downsample_0
        )
        model_layer3_0_downsample_0 = None
        model_layer3_0_bn2 += model_layer3_0_downsample_1
        iadd_4 = model_layer3_0_bn2
        model_layer3_0_bn2 = model_layer3_0_downsample_1 = None
        model_layer3_0_relu_1 = self.model_layer3_0_relu(iadd_4)
        iadd_4 = None
        model_layer3_1_conv1 = self.model_layer3_1_conv1(model_layer3_0_relu_1)
        model_layer3_1_bn1 = self.model_layer3_1_bn1(model_layer3_1_conv1)
        model_layer3_1_conv1 = None
        model_layer3_1_relu = self.model_layer3_1_relu(model_layer3_1_bn1)
        model_layer3_1_bn1 = None
        model_layer3_1_conv2 = self.model_layer3_1_conv2(model_layer3_1_relu)
        model_layer3_1_relu = None
        model_layer3_1_bn2 = self.model_layer3_1_bn2(model_layer3_1_conv2)
        model_layer3_1_conv2 = None
        model_layer3_1_bn2 += model_layer3_0_relu_1
        iadd_5 = model_layer3_1_bn2
        model_layer3_1_bn2 = model_layer3_0_relu_1 = None
        model_layer3_1_relu_1 = self.model_layer3_1_relu(iadd_5)
        iadd_5 = None
        model_layer4_0_conv1 = self.model_layer4_0_conv1(model_layer3_1_relu_1)
        model_layer4_0_bn1 = self.model_layer4_0_bn1(model_layer4_0_conv1)
        model_layer4_0_conv1 = None
        model_layer4_0_relu = self.model_layer4_0_relu(model_layer4_0_bn1)
        model_layer4_0_bn1 = None
        model_layer4_0_conv2 = self.model_layer4_0_conv2(model_layer4_0_relu)
        model_layer4_0_relu = None
        model_layer4_0_bn2 = self.model_layer4_0_bn2(model_layer4_0_conv2)
        model_layer4_0_conv2 = None
        model_layer4_0_downsample_0 = self.model_layer4_0_downsample_0(
            model_layer3_1_relu_1
        )
        model_layer3_1_relu_1 = None
        model_layer4_0_downsample_1 = self.model_layer4_0_downsample_1(
            model_layer4_0_downsample_0
        )
        model_layer4_0_downsample_0 = None
        model_layer4_0_bn2 += model_layer4_0_downsample_1
        iadd_6 = model_layer4_0_bn2
        model_layer4_0_bn2 = model_layer4_0_downsample_1 = None
        model_layer4_0_relu_1 = self.model_layer4_0_relu(iadd_6)
        iadd_6 = None
        model_layer4_1_conv1 = self.model_layer4_1_conv1(model_layer4_0_relu_1)
        model_layer4_1_bn1 = self.model_layer4_1_bn1(model_layer4_1_conv1)
        model_layer4_1_conv1 = None
        model_layer4_1_relu = self.model_layer4_1_relu(model_layer4_1_bn1)
        model_layer4_1_bn1 = None
        model_layer4_1_conv2 = self.model_layer4_1_conv2(model_layer4_1_relu)
        model_layer4_1_relu = None
        model_layer4_1_bn2 = self.model_layer4_1_bn2(model_layer4_1_conv2)
        model_layer4_1_conv2 = None
        model_layer4_1_bn2 += model_layer4_0_relu_1
        iadd_7 = model_layer4_1_bn2
        model_layer4_1_bn2 = model_layer4_0_relu_1 = None
        model_layer4_1_relu_1 = self.model_layer4_1_relu(iadd_7)
        iadd_7 = None
        model_avgpool = self.model_avgpool(model_layer4_1_relu_1)
        model_layer4_1_relu_1 = None
        flatten = torch.flatten(model_avgpool, 1)
        model_avgpool = None
        model_fc = self.model_fc(flatten)
        flatten = None
        return (model_fc,)
