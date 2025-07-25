# CNN Models with SYCL Convolution Integration

This directory includes PyTorch CNN model definitions (VGG16, ResNet101, InceptionV4) where the standard `nn.Conv2d` layers are replaced with custom convolution layers implemented using SYCL kernels.

## Files

- `vgg16.py` — VGG16 architecture using SYCL-accelerated conv layers.
- `resnet101.py` — ResNet101 variant using SYCL-accelerated conv layers.
- `inceptionv4.py` — InceptionV4 with customizable SYCL convs.

## Supported Algorithms

Each convolution layer can use one of the following algorithms by specifying `algo=`:

- `"smm"`
- `"kn2row"`
- `"im2col"`
- `"direct"`
- `"depthwise"`

## Example Usage

```python
# Example inside ConvLayer module
ConvLayer(..., algo="im2col")
```

## Run

```bash
python vgg16.py
python resnet101.py
python inceptionv4.py
```

Each script reports per-layer execution time and total inference latency.
