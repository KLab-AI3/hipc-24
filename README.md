# Exploring Algorithmic Design Choices for Low Latency CNN Deployment - HIPC 2024

This repository accompanies the paper **"Exploring Algorithmic Design Choices for Low Latency CNN Deployment"**, presented at **HIPC 2024**.  
DOI: [10.1109/HiPC62374.2024.00017](https://doi.org/10.1109/HiPC62374.2024.00017)

It provides:

- SYCL-based implementations of five convolution algorithms using Intel oneAPI DPC++.
- Three CNN models (**VGG16**, **ResNet101**, **InceptionV4**) where all `Conv2d` layers are replaced with custom SYCL-based implementations via `ctypes`.
- Benchmarks and execution time tracking for each convolution layer.

---

## 📌 Algorithms Implemented

| Algorithm   | Description                          | Shared Library  |
|-------------|--------------------------------------|------------------|
| `smm`       | Scalar Matrix Multiplication         | `smm_conv.so`    |
| `kn2row`    | Kernel to Row flattening             | `kn2row.so`      |
| `im2col`    | Image to Column flattening           | `im2col.so`      |
| `direct`    | Naive nested-loop convolution        | `direct.so`      |
| `depthwise` | Depthwise separable convolution      | `depthwise.so`   |

---

## 🔧 Prerequisites

### ✅ Python Environment

Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv sycl_env
source sycl_env/bin/activate
```

Then install Python dependencies:

```bash
pip install -r requirements.txt
```

> This installs `torch`, `torchvision`, and other required packages.

`requirements.txt` includes:

```
torch>=2.4.0
torchvision>=0.15.0
```

---

### ✅ HPC Environment (V100 GPU Recommended)

Before compiling SYCL kernels with oneAPI and targeting CUDA backend, make sure the following modules or system libraries are available:

```bash
module load SYCL/2024.0.1.46
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
```

> These provide Intel DPC++ compiler, CUDA libraries, and cuDNN support for GPU execution.
---

## ⚙️ Build SYCL Kernels

Each algorithm is written in its own `.cpp` file. You can compile any one of them using:

```bash
cd sycl_kernels
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
     -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 \
     SMM.cpp -o ../smm_conv.so -fPIC -shared -lm
```
> `sm_70` is the architecture for NVIDIA V100. Output should be a `.so` shared library.

Repeat for: `Direct.cpp`, `Kn2row.cpp`, `Im2col.cpp`, `Depthwise.cpp`


---

## 🧠 SYCL-Accelerated CNN Models

All `nn.Conv2d` layers are replaced by a `ConvLayer` that internally calls SYCL `.so` libraries via `ctypes`.

The following models are implemented and support all five convolution algorithms:

- ✅ VGG16
- ✅ ResNet101
- ✅ InceptionV4

You can choose the algorithm via command line argument:

### 🔧 Supported Algorithms

```python
algo = "smm"         # Scalar Matrix Multiplication
algo = "kn2row"      # Kernel to Row flattening
algo = "im2col"      # Image to Column flattening
algo = "direct"      # Naive nested-loop
algo = "depthwise"   # Per-channel depthwise
```


### ✅ Supported Models

| Model      | Script File              |
|------------|--------------------------|
| VGG16      | `models/vgg16_sycl.py`   |
| ResNet101  | `models/resnet101_sycl.py` |
| InceptionV4| `models/inceptionv4_sycl.py` |

### 🔧 Algorithm Switch

Set the `--algo` flag to select the convolution method:

```bash
python models/vgg16_sycl.py --algo smm
python models/resnet101_sycl.py --algo im2col
python models/inceptionv4_sycl.py --algo kn2row
```

---

### 🚀 Run SYCL Kernel Executables (Optional)

You can also run the `.out` binaries directly (for kernel testing only):

```bash
./SMM.out
```

These log timing results to a CSV file like:

```
smm_result.csv
```

---


## 📁 Project Structure

```
.
├── sycl_kernels/             # 🔧 SYCL kernel implementations
│   ├── SMM.cpp               # Scalar Matrix Multiplication
│   ├── Kn2row.cpp            # Kernel to Row
│   ├── Im2col.cpp            # Image to Column
│   ├── Direct.cpp            # Naive loop
│   ├── Depthwise.cpp         # Depthwise convolution
│   └── README.md             # Kernel compilation guide
│
├── models/                   # 🧠 CNN model implementations
│   ├── vgg16_sycl.py              # VGG16 with dynamic SYCL conv
│   ├── resnet101_sycl.py          # ResNet101 with dynamic SYCL conv
│   ├── inceptionv4_sycl.py        # InceptionV4 with dynamic SYCL conv
│   └── README.md             # Usage instructions
│
├── requirements.txt          # Python dependencies
├── README.md                 # You are here
```

---

## 📚 Citation

If you use this work, please cite:

> **"Exploring Algorithmic Design Choices for Low Latency CNN Deployment"**  
> Changxin Li, Sanmukh Kuppannagari — Case Western Reserve University

```bibtex
@INPROCEEDINGS{10884187,
  author={Li, Changxin and Kuppannagari, Sanmukh},
  booktitle={2024 IEEE 31st International Conference on High Performance Computing, Data, and Analytics (HiPC)}, 
  title={Exploring Algorithmic Design Choices for Low Latency CNN Deployment}, 
  year={2024},
  pages={78-88},
  doi={10.1109/HiPC62374.2024.00017}
}
```
