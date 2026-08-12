# Fast WTConv

A high-performance implementation of Wavelet Convolution (WTConv) layers with optimized backends for CUDA, Apple Metal (MPS), and Triton.

## Overview

`fast_wtconv` provides a drop-in replacement for WTConv layers that significantly accelerates training and inference.

The CUDA code is the official implementation accompanying the paper [*Fast and Memory-Efficient
Wavelet Convolutions via I/O-Aware Reformulation*](https://arxiv.org/abs/2608.10805) and implements
its fused, I/O-aware reformulation of WTConv.

## Features

- **Multi-Backend Support**:
  - **CUDA**: Fused CUDA kernels for NVIDIA GPUs. Supports fp32, fp16, and bf16.
  - **Metal (MPS)**: Optimized Metal shaders for Apple Silicon (M1/M2/M3). Supports fp32 and fp16.
  - **Triton**: Pure Triton implementation for portability and high performance without CUDA dependencies. Supports fp32, fp16, and bf16.
- **Seamless Integration**: Matches the API of the original `WTConv2d` for easy integration into existing models.


## Performance

Speedup compared to the original WTConv implementation (Kernel Size: 5, FP32):

| Platform | Hardware | Speedup |
|----------|----------|---------|
| **CUDA**  | RTX A6000 | ~2.9x |
| **Triton** | RTX A6000 | ~3.0x |
| **Metal** | Apple M3 | ~2.3x |

## Installation

Ensure you have the necessary dependencies installed:
- PyTorch
- Triton (for the Triton backend)

> [!NOTE]
> All implementations use JIT (Just-In-Time) compilation. For the CUDA backend, you must have `nvcc` (NVIDIA CUDA Compiler) installed and available in your system PATH for it to work.

### CUDA kernel sizes

`kernel_size` must be odd. Every odd size up to the build ceiling is compiled, and compile time grows
superlinearly in the kernel size, so the default ceiling is **7** (a ~30 s first-use build). For a
larger kernel, set `HAAR_MAX_K` to an odd value up to 29 and rerun:

```sh
HAAR_MAX_K=9 python your_script.py
```

Each ceiling is cached as a separate build, so it is a one-time cost. Asking for a `kernel_size` above
the ceiling raises an error naming the value to rebuild with, before anything is compiled.

## Usage

### CUDA Backend

```python
import torch
from fast_wtconv.wtconv_cuda import WTConv2d

# in_channels, out_channels, kernel_size, stride, wt_levels
layer = WTConv2d(64, 64, kernel_size=5, wt_levels=2).cuda()

x = torch.randn(1, 64, 224, 224).cuda()
output = layer(x)
```

### Metal (MPS) Backend

```python
import torch
from fast_wtconv.wtconv_metal import WTConv2d

layer = WTConv2d(64, 64, kernel_size=5, wt_levels=2).to('mps')

x = torch.randn(1, 64, 224, 224).to('mps')
output = layer(x)
```

### Triton Backend

If you prefer the pure Triton implementation (e.g., for AMD GPUs or specific performance profiles), use the `wtconv_triton` module.

```python
import torch
from fast_wtconv.wtconv_triton import WTConv2d as WTConv2dTriton

# Initialize Triton layer
layer = WTConv2dTriton(64, 64, kernel_size=5, wt_levels=2).cuda()

# Forward pass
x = torch.randn(1, 64, 224, 224).cuda()
output = layer(x)
```

## Directory Structure

- `wtconv_cuda.py`: `WTConv2d` on the fused CUDA kernels.
- `wtconv_metal.py`: `WTConv2d` on the Metal (MPS) kernels.
- `wtconv_triton.py`: Pure Triton implementation.
- `cuda_haar/`: CUDA kernel implementations and bindings.
- `metal_haar/`: Metal shader implementations and bindings.
- `triton_haar/`: Triton kernel implementations.

## Notes

- CUDA weight gradients accumulate through fp32 atomics, so they are not bitwise reproducible run to
  run (neither is cuDNN's depthwise weight gradient, which they replace).
- As in the original implementation, `stride > 1` is applied as average pooling on the output, and odd
  spatial sizes are zero-padded at each decomposition level and cropped back on reconstruction.

> [!NOTE]
> The CUDA backend was replaced with a fused, I/O-aware implementation. At the same time, the former
> auto-detecting `wtconv.py` was split into `wtconv_cuda.py` and `wtconv_metal.py`, one per backend.
> The previous CUDA backend and the combined entry point are at commit `1e9a25c`.