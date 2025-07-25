import torch
import torch.nn as nn
import time
from torch.autograd import Function
import ctypes
import numpy as np
from torchvision.models.resnet import Bottleneck

# === SYCL Wrapper using ctypes ===
class SyclConv2d(Function):
    libname = "./smm_conv.so"  # default

    @staticmethod
    def forward(ctx, input, kernel, stride, padding):
        input = input.contiguous()
        kernel = kernel.contiguous()
        batch, in_ch, in_h, in_w = input.shape
        out_ch, _, kh, kw = kernel.shape

        if isinstance(padding, tuple): padding = padding[0]
        if isinstance(stride, tuple): stride = stride[0]

        out_h = (in_h - kh + 2 * padding) // stride + 1
        out_w = (in_w - kw + 2 * padding) // stride + 1

        output = torch.zeros((batch, out_ch, out_h, out_w), device=input.device)

        lib = ctypes.PyDLL(SyclConv2d.libname)
        lib_func = getattr(lib, SyclConv2d.libname.split('/')[-1].split('.')[0] + '_conv')
        lib_func.restype = None

        input_np = input.view(-1).detach().cpu().numpy().astype(np.float32)
        kernel_np = kernel.view(-1).detach().cpu().numpy().astype(np.float32)
        output_np = output.view(-1).detach().cpu().numpy().astype(np.float32)

        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        kernel_ptr = kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib_func(input_ptr, kernel_ptr, output_ptr,
                 ctypes.c_int(in_h), ctypes.c_int(in_w),
                 ctypes.c_int(in_ch), ctypes.c_int(kh), ctypes.c_int(kw),
                 ctypes.c_int(out_ch), ctypes.c_int(stride), ctypes.c_int(padding))

        return torch.tensor(output_np).view(output.shape).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

def smm_conv2d(input, kernel, stride=1, padding=0):
    SyclConv2d.libname = "./smm_conv.so"; return SyclConv2d.apply(input, kernel, stride, padding)

def kn2row_conv2d(input, kernel, stride=1, padding=0):
    SyclConv2d.libname = "./kn2row.so"; return SyclConv2d.apply(input, kernel, stride, padding)

def im2col_conv2d(input, kernel, stride=1, padding=0):
    SyclConv2d.libname = "./im2col.so"; return SyclConv2d.apply(input, kernel, stride, padding)

def direct_conv2d(input, kernel, stride=1, padding=0):
    SyclConv2d.libname = "./direct.so"; return SyclConv2d.apply(input, kernel, stride, padding)

def depthwise_conv2d(input, kernel, stride=1, padding=0):
    SyclConv2d.libname = "./depthwise.so"; return SyclConv2d.apply(input, kernel, stride, padding)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, algo="smm"):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.stride = stride
        self.padding = padding
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.algo = algo

    def forward(self, x):
        start = time.time()
        if self.algo == "smm":
            x = smm_conv2d(x, self.kernel, self.stride, self.padding)
        elif self.algo == "kn2row":
            x = kn2row_conv2d(x, self.kernel, self.stride, self.padding)
        elif self.algo == "im2col":
            x = im2col_conv2d(x, self.kernel, self.stride, self.padding)
        elif self.algo == "direct":
            x = direct_conv2d(x, self.kernel, self.stride, self.padding)
        elif self.algo == "depthwise":
            x = depthwise_conv2d(x, self.kernel, self.stride, self.padding)
        else:
            raise ValueError(f"Unknown algorithm: {self.algo}")
        print(f"{self.algo} layer time: {time.time() - start:.6f}s")
        x = self.bn(x)
        return self.relu(x)

# BasicBlock variant using SYCL conv
class SyclBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, algo="smm"):
        super().__init__()
        self.conv1 = ConvLayer(inplanes, planes, (1,1), 1, 0, algo)
        self.conv2 = ConvLayer(planes, planes, (3,3), stride, 1, algo)
        self.conv3 = ConvLayer(planes, planes * self.expansion, (1,1), 1, 0, algo)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# Assemble ResNet101
class ResNet101_Sycl(nn.Module):
    def __init__(self, num_classes=1000, algo="smm"):
        super().__init__()
        self.inplanes = 64
        self.conv1 = ConvLayer(3, 64, (7,7), 2, 3, algo)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, algo)
        self.layer2 = self._make_layer(128, 4, algo, stride=2)
        self.layer3 = self._make_layer(256, 23, algo, stride=2)
        self.layer4 = self._make_layer(512, 3, algo, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, planes, blocks, algo, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                ConvLayer(self.inplanes, planes * 4, (1,1), stride, 0, algo)
            )
        layers = [SyclBottleneck(self.inplanes, planes, stride, downsample, algo)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(SyclBottleneck(self.inplanes, planes, algo=algo))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet101_Sycl(algo="smm").to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    start = time.time()
    y = model(x)
   
