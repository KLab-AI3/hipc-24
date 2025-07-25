# 🧠 CNN Models with SYCL-Accelerated Convolution

This folder contains PyTorch implementations of three CNN models — **VGG16**, **ResNet101**, and **InceptionV4** — where all `nn.Conv2d` layers have been **replaced** with custom `ConvLayer` modules.

Each `ConvLayer` dynamically dispatches a SYCL-based convolution algorithm (e.g., SMM, Kn2row, Im2col, Direct, or Depthwise) via `ctypes` to call precompiled `.so` shared libraries.

---

## 🔍 Available Models

| File              | Model        | Description                            |
|-------------------|--------------|----------------------------------------|
| `vgg16.py`        | VGG16        | Classic VGG16 with SYCL conv layers    |
| `resnet101.py`    | ResNet101    | Deep ResNet with SYCL conv replacement |
| `inceptionv4.py`  | InceptionV4  | Multi-branch Inception with SYCL conv  |

---

## ⚙️ Runtime Parameters

Each script accepts a command-line argument `--algo` to select the convolution algorithm at runtime:

```bash
--algo smm        # Scalar Matrix Multiplication
--algo kn2row     # Kernel flattening
--algo im2col     # Image to column
--algo direct     # Naive loop-based
--algo depthwise  # Per-channel depthwise
```

---

## 🚀 Example Usage

Run VGG16 with the SMM algorithm:

```bash
python vgg16.py --algo smm
```

Run ResNet101 with Kn2row:

```bash
python resnet101.py --algo kn2row
```

Run InceptionV4 with Depthwise:

```bash
python inceptionv4.py --algo depthwise
```

---

## 📦 Output

- Execution time (per layer)
- Total runtime of the model
- Output tensor shape

Sample output:

```text
[ConvLayer] Layer: conv3_1 | Algo: smm | Time: ...s
[ConvLayer] Layer: conv3_2 | Algo: smm | Time: ...s
Total execution time: 0.221s
Output shape: torch.Size([8, 1000])
```

> NOTE: Make sure the corresponding `.so` file (e.g., `smm_conv.so`) is in the project root directory.

---

## 🔁 Extend or Replace

To extend with more models, follow the same structure:

- Replace `nn.Conv2d` with `ConvLayer(...)`
- Use `algo` argument to dynamically load the corresponding shared library

---

## 🛠 Dependencies

All models require:

- PyTorch ≥ 2.4
- CUDA-capable GPU (e.g., V100)
- Precompiled SYCL `.so` files (e.g., `smm_conv.so`)

```bash
pip install -r ../requirements.txt
```
