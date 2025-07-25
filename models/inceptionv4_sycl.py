import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes
import numpy as np
import time

# ========== SYCL Convolution Wrapper ==========
class SyclConv2d(torch.autograd.Function):
    libname = "./smm_conv.so"  # default

    @staticmethod
    def forward(ctx, input, kernel, stride, padding):
        input = input.contiguous()
        kernel = kernel.contiguous()

        input_height, input_width = input.shape[-2:]
        kernel_height, kernel_width = kernel.shape[-2:]
        input_channels = input.shape[1]
        output_channels = kernel.shape[0]

        if isinstance(stride, tuple): stride = stride[0]
        if isinstance(padding, tuple): padding = padding[0]

        output_height = (input_height - kernel_height + 2 * padding) // stride + 1
        output_width = (input_width - kernel_width + 2 * padding) // stride + 1

        output = torch.zeros((input.shape[0], output_channels, output_height, output_width), device=input.device)

        lib = ctypes.PyDLL(SyclConv2d.libname)
        conv_func = lib.smm_conv
        conv_func.restype = None

        input_np = input.detach().cpu().numpy().astype(np.float32).flatten()
        kernel_np = kernel.detach().cpu().numpy().astype(np.float32).flatten()
        output_np = output.detach().cpu().numpy().astype(np.float32).flatten()

        conv_func(
            input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(input_height),
            ctypes.c_int(input_width),
            ctypes.c_int(input_channels),
            ctypes.c_int(kernel_height),
            ctypes.c_int(kernel_width),
            ctypes.c_int(output_channels),
            ctypes.c_int(stride),
            ctypes.c_int(padding)
        )

        return torch.tensor(output_np, device=input.device).view(output.shape)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()

# ========== Algorithm Switcher ==========
def select_conv(algo):
    def f(input, kernel, stride, padding):
        if algo == "smm":
            SyclConv2d.libname = "./smm_conv.so"
        elif algo == "kn2row":
            SyclConv2d.libname = "./kn2row.so"
        elif algo == "im2col":
            SyclConv2d.libname = "./im2col.so"
        elif algo == "direct":
            SyclConv2d.libname = "./direct.so"
        elif algo == "depthwise":
            SyclConv2d.libname = "./depthwise.so"
        else:
            raise ValueError(f"Unsupported algo {algo}")
        return SyclConv2d.apply(input, kernel, stride, padding)
    return f

# ========== Custom Conv Layer ==========
class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, algo="smm"):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_c, in_c, k, k))
        self.stride = s
        self.padding = p
        self.algo = algo
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        start = time.time()
        x = select_conv(self.algo)(x, self.kernel, self.stride, self.padding)
        print(f"{self.algo.upper()} time: {time.time() - start:.6f}s")
        return self.relu(self.bn(x))

# ========== Minimal InceptionV4 Stem Block ==========
class Stem(nn.Module):
    def __init__(self, algo="smm"):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, k=3, s=2, p=0, algo=algo)    # 149x149
        self.conv2 = ConvLayer(32, 32, k=3, s=1, p=0, algo=algo)   # 147x147
        self.conv3 = ConvLayer(32, 64, k=3, s=1, p=1, algo=algo)   # 147x147
        self.pool = nn.MaxPool2d(3, 2)                             # 73x73

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pool(x)

# ========== Simple Inception-A Block ==========
class InceptionA(nn.Module):
    def __init__(self, in_c, algo="smm"):
        super().__init__()
        self.branch1 = ConvLayer(in_c, 64, 1, algo=algo)
        self.branch2 = nn.Sequential(
            ConvLayer(in_c, 48, 1, algo=algo),
            ConvLayer(48, 64, 5, p=2, algo=algo)
        )
        self.branch3 = nn.Sequential(
            ConvLayer(in_c, 64, 1, algo=algo),
            ConvLayer(64, 96, 3, p=1, algo=algo),
            ConvLayer(96, 96, 3, p=1, algo=algo)
        )
        self.pool_proj = ConvLayer(in_c, 32, 1, algo=algo)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.pool_proj(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([branch1, branch2, branch3, branch4], 1)

# ========== InceptionV4 ==========
class InceptionV4_Sycl(nn.Module):
    def __init__(self, num_classes=1000, algo="smm"):
        super().__init__()
        self.stem = Stem(algo=algo)
        self.incepA1 = InceptionA(64, algo=algo)
        self.incepA2 = InceptionA(256, algo=algo)
        self.incepA3 = InceptionA(256, algo=algo)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incepA1(x)
        x = self.incepA2(x)
        x = self.incepA3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ========== Run Test ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV4_Sycl(algo="smm").to(device)
    x = torch.randn(4, 3, 299, 299).to(device)
    with torch.no_grad():
        out = model(x)
    
