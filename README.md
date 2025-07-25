# Exploring Algorithmic Design Choices for Low Latency CNN Deployment-HIPC-2024
This repo contains code associated with the paper: "Exploring Algorithmic Design Choices for Low Latency CNN Deployment". It was presented at HIPC 2024 and the text can be found at: https://doi.org/10.1109/HiPC62374.2024.00017.
It contains the SYCL-based implementations of five convolution algorithms (IM2COL, KN2ROW, SMM, Direct, and Depthwise) using Intel oneAPI DPC++ (SYCL) and their integration into three popular CNN models: VGG16, ResNet101, and Inception V4. 
The goal is to evaluate and compare different algorithmic strategies for accelerating CNN convolution layers on NVIDIA GPUs.

---

## 📌 Algorithms Implemented

1. **Direct Convolution** — Naive nested-loop implementation.
2. **Depthwise Convolution** — One kernel per input channel.
3. **Im2col** — Column-wise patch flattening + matrix multiplication.
4. **Kn2row** — Kernel flattening technique.
5. **SMM (Scalar Matrix Multiplication)** — Sparse-aware direct multiplication, baseline for unstructured convolution.

---

## 🔧 Prerequisites

### ✅ Installation via `requirements.txt` (Python)

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

---

### ✅ System Requirements (HPC with NVIDIA V100)

Before compiling SYCL kernels with oneAPI and targeting CUDA backend, make sure the following modules or system libraries are available:

```bash
module load SYCL/2024.0.1.46
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
```

> These provide Intel DPC++ compiler, CUDA libraries, and cuDNN support for GPU execution.

---

## 🧠 CNN Models with SYCL Acceleration

We replace PyTorch's native `nn.Conv2d` with custom `ConvLayer`, which internally calls SYCL-accelerated convolution kernels using `ctypes`.

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

### 🚀 Example Usage

Run VGG16 using `smm` convolution:

```bash
python models/vgg16.py --algo smm
```

Run ResNet101 using `im2col`:

```bash
python models/resnet101.py --algo im2col
```

Run InceptionV4 using `kn2row`:

```bash
python models/inceptionv4.py --algo kn2row
```

Each convolution layer will print its execution time (in seconds), and the model will print the total execution time at the end.

---

## ⚙️ Build Instructions

Each algorithm is written in its own `.cpp` file. You can compile any one of them using:

```bash
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
     -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 \
     sycl_kernels/SMM.cpp -o smm_conv.so -fPIC -shared -lm
```

Replace `SMM.cpp` with any of the following:

- `Direct.cpp`
- `Depthwise.cpp`
- `Im2col.cpp`
- `Kn2row.cpp`

> `sm_70` is the architecture for NVIDIA V100. Output should be a `.so` shared library.

---

## 🚀 Run SYCL Kernel Executables (Optional)

You can also run the `.out` binaries directly (for kernel testing only):

```bash
./SMM.out
```

These log timing results to a CSV file like:

```
smm_result.csv
```

---

## 📁 Project File Structure

```bash
.
├── sycl_kernels/                     # 🔧 SYCL convolution kernel implementations
│   ├── Direct.cpp                    # Direct convolution
│   ├── Depthwise.cpp                 # Depthwise convolution
│   ├── Im2col.cpp                    # Im2col + GEMM
│   ├── Kn2row.cpp                    # Kn2row (GEMM variant)
│   ├── SMM.cpp                       # Scalar Matrix Multiplication baseline
│   └── README.md                     # Compilation & usage for SYCL kernels
│
├── models/                           # 🧠 CNN models with swapped conv layers
│   ├── vgg16.py                      # VGG16 with replaceable SYCL conv layers
│   ├── resnet101.py                  # ResNet101 with replaceable SYCL conv layers
│   ├── inceptionv4.py                # InceptionV4 with replaceable SYCL conv layers
│   └── README.md                     # Usage instructions for models
│                        # PyTorch baseline test script
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview & installation guide
```

---

## 📊 Output Format

CSV files are generated for each algorithm with format:

```csv
IC,OC,H,W,KH,KW,pad,stride,time_us
```

---

## 📚 Citation

This work is part of the paper:

> **"Exploring Algorithmic Design Choices for Low Latency CNN Deployment"**  
> Changxin Li,  Prof. Sanmukh Kuppannagari — Case Western Reserve University

Please cite appropriately if used in academic research.
> @INPROCEEDINGS{10884187,
  author={Li, Changxin and Kuppannagari, Sanmukh},
  booktitle={2024 IEEE 31st International Conference on High Performance Computing, Data, and Analytics (HiPC)}, 
  title={Exploring Algorithmic Design Choices for Low Latency CNN Deployment}, 
  year={2024},
  volume={},
  number={},
  pages={78-88},
  keywords={Machine learning algorithms;Convolution;Graphics processing units;Parallel processing;Prediction algorithms;Hardware;Data models;Convolutional neural networks;Low latency communication;Standards;Algorithmic Design Choices;Convolution Algorithms;Vision Models;Performance Portability},
  doi={10.1109/HiPC62374.2024.00017}}
---

## ✅ Installation Summary

### 🐍 Step 1: Set Up Python Environment

```bash
python3 -m venv sycl_env
source sycl_env/bin/activate
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch>=2.4.0
torchvision>=0.15.0
```

---

### ⚙️ Step 2: Load System Modules (For V100 HPC)

```bash
module load SYCL/2024.0.1.46
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
```

---

### 🛠️ Step 3: Compile SYCL Kernels

```bash
cd sycl_kernels
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
     -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 \
     SMM.cpp -o ../smm_conv.so -fPIC -shared -lm
```

Repeat for other algorithms: `Kn2row.cpp`, `Im2col.cpp`, etc.

---

### 🧠 Step 4: Run Model with Desired Algorithm

```bash
cd models
python vgg16.py --algo smm
```

Outputs total execution time and per-layer timing.

