# 🔧 SYCL Kernel Implementations

This directory contains SYCL-based GPU convolution kernel implementations targeting NVIDIA GPUs (V100). Each file represents a different algorithmic strategy for performing 2D convolution.

## Files

- `Direct.cpp` — Basic nested-loop direct convolution.
- `Depthwise.cpp` — Depthwise separable convolution kernel.
- `Im2col.cpp` — Converts input to column matrix followed by GEMM.
- `Kn2row.cpp` — Similar to im2col but reshapes kernel.
- `SMM.cpp` — Scalar matrix multiplication.
  
## Compile Example

To compile a kernel (e.g., SMM):

```bash
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
     -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 \
     SMM.cpp -o SMM.out -lm
```

## Output

Each implementation outputs a CSV file with the following format:

```csv
IC,OC,H,W,KH,KW,pad,stride,time_us
```

Use these logs to compare timing across different strategies.
