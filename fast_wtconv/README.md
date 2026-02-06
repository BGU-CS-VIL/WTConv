# Fast WTConv

A high-performance implementation of Wavelet Convolution (WTConv) layers with optimized backends for CUDA, Apple Metal (MPS), and Triton.

## Overview

`fast_wtconv` provides a drop-in replacement for WTConv layers that significantly accelerates training and inference.

## Features

- **Multi-Backend Support**:
  - **CUDA**: Optimized CUDA kernels for NVIDIA GPUs. Supports fp32, fp16, and bf16.
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

## Usage

### Standard Usage (Auto-Detect)

The standard `WTConv2d` class automatically detects your device (CUDA or MPS) and uses the appropriate optimized kernel.

```python
import torch
from fast_wtconv.wtconv import WTConv2d

# Initialize layer
# in_channels, out_channels, kernel_size, stride, wt_levels
layer = WTConv2d(64, 64, kernel_size=5, wt_levels=2)

# Move to device (CUDA or MPS)
device = 'cuda' if torch.cuda.is_available() else 'mps'
layer = layer.to(device)

# Forward pass
x = torch.randn(1, 64, 224, 224).to(device)
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

- `wtconv.py`: Main entry point with auto-backend selection (CUDA/Metal).
- `wtconv_triton.py`: Pure Triton implementation.
- `cuda_haar/`: CUDA kernel implementations and bindings.
- `metal_haar/`: Metal shader implementations and bindings.
- `triton_haar/`: Triton kernel implementations.
