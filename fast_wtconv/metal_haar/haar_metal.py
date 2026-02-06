"""
Metal Haar Wavelet Transform Library

Provides autograd-compatible Haar wavelet transforms on Apple Metal:
- haar2d / ihaar2d: Single-level transform
- haar2d_double: 2-level cascade
- haar2d_triple: 3-level cascade  
- haar2d_quad: 4-level cascade
- haar2d_quint: 5-level cascade

Tensors must be on MPS device for GPU acceleration.

Usage:
    from metal_haar import haar2d, ihaar2d, haar2d_double
    
    # Single level
    x = torch.randn(B, C, H, W, device='mps')
    coeffs = haar2d(x)     # (B, C, H, W) -> (B, C, 4, H/2, W/2)
    recon = ihaar2d(coeffs, (H, W))  # (B, C, 4, H/2, W/2) -> (B, C, H, W)
    
    # Multi-level cascade
    l1, l2 = haar2d_double(x)
"""

import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
from pathlib import Path
from typing import Tuple, Optional

# =============================================================================
# JIT Compile Metal Extension
# =============================================================================

_module = None

def _get_module():
    global _module
    if _module is None:
        src_dir = Path(__file__).parent
        print("Compiling Metal Haar kernels...")
        _module = load(
            name='haar_metal_cpp',
            sources=[str(src_dir / 'haar.mm')],
            extra_cflags=['-std=c++17', '-O3'],
            extra_ldflags=['-framework', 'Metal', '-framework', 'Foundation'],
            verbose=False
        )
        _module.set_metal_source_path(str(src_dir))
        print("Done.")
    return _module


# =============================================================================
# Single Level Haar Transform
# =============================================================================

class HaarTransform(Function):
    """Single-level 2D Haar wavelet transform with autograd support."""
    
    @staticmethod
    def forward(ctx, x):
        assert x.is_mps, "Input must be on MPS device"
        assert x.dtype in (torch.float32, torch.float16), "Only float32/float16 supported"
        assert x.dim() == 4, "Input must be (B, C, H, W)"
        
        B, C, H, W = x.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        
        # Allocate output on MPS
        output = torch.empty((B, C, 4, H2, W2), dtype=x.dtype, device=x.device)
        
        _get_module().haar2d_forward(x.contiguous(), output)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = x.dtype
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        H, W = ctx.shape_hw
        B, C = grad_output.shape[:2]
        
        grad_input = torch.zeros((B, C, H, W), dtype=ctx.dtype, device=grad_output.device)
        
        _get_module().haar2d_backward(grad_output.contiguous(), grad_input)
        
        return grad_input


class InverseHaarTransform(Function):
    """Single-level inverse 2D Haar wavelet transform with autograd support."""
    
    @staticmethod
    def forward(ctx, x, output_size=None):
        assert x.is_mps, "Input must be on MPS device"
        assert x.dtype in (torch.float32, torch.float16), "Only float32/float16 supported"
        assert x.dim() == 5, "Input must be (B, C, 4, H2, W2)"
        
        B, C, _, H2, W2 = x.shape
        
        if output_size is not None:
            H, W = output_size
        else:
            H, W = H2 * 2, W2 * 2
        
        output = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
        
        _get_module().haar2d_inverse(x.contiguous(), output)
        
        ctx.shape_h2w2 = (H2, W2)
        ctx.dtype = x.dtype
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        H2, W2 = ctx.shape_h2w2
        B, C = grad_output.shape[:2]
        
        grad_input = torch.empty((B, C, 4, H2, W2), dtype=ctx.dtype, device=grad_output.device)
        
        _get_module().haar2d_inverse_backward(grad_output.contiguous(), grad_input)
        
        return grad_input, None


# =============================================================================
# Multi-Level Cascade Transforms
# =============================================================================

class HaarDoubleTransform(Function):
    """2-level cascade Haar transform with autograd support."""
    
    @staticmethod
    def forward(ctx, x):
        assert x.is_mps, "Input must be on MPS device"
        assert x.dim() == 4
        
        B, C, H, W = x.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        H4, W4 = (H + 3) // 4, (W + 3) // 4
        
        level1 = torch.empty((B, C, 4, H2, W2), dtype=x.dtype, device=x.device)
        level2 = torch.empty((B, C, 4, H4, W4), dtype=x.dtype, device=x.device)
        
        _get_module().haar2d_double_forward(x.contiguous(), level1, level2)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = x.dtype
        
        return level1, level2
    
    @staticmethod
    def backward(ctx, grad_level1, grad_level2):
        H, W = ctx.shape_hw
        B, C = grad_level2.shape[:2]
        
        grad_input = torch.zeros((B, C, H, W), dtype=ctx.dtype, device=grad_level2.device)
        
        _get_module().haar2d_double_backward(grad_level1.contiguous(), grad_level2.contiguous(), grad_input)
        
        return grad_input


class HaarTripleTransform(Function):
    """3-level cascade Haar transform with autograd support."""
    
    @staticmethod
    def forward(ctx, x):
        assert x.is_mps, "Input must be on MPS device"
        assert x.dim() == 4
        
        B, C, H, W = x.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        H4, W4 = (H + 3) // 4, (W + 3) // 4
        H8, W8 = (H + 7) // 8, (W + 7) // 8
        
        level1 = torch.empty((B, C, 4, H2, W2), dtype=x.dtype, device=x.device)
        level2 = torch.empty((B, C, 4, H4, W4), dtype=x.dtype, device=x.device)
        level3 = torch.empty((B, C, 4, H8, W8), dtype=x.dtype, device=x.device)
        
        _get_module().haar2d_triple_forward(x.contiguous(), level1, level2, level3)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = x.dtype
        
        return level1, level2, level3
    
    @staticmethod
    def backward(ctx, grad_level1, grad_level2, grad_level3):
        H, W = ctx.shape_hw
        B, C = grad_level3.shape[:2]
        
        grad_input = torch.zeros((B, C, H, W), dtype=ctx.dtype, device=grad_level3.device)
        
        _get_module().haar2d_triple_backward(
            grad_level1.contiguous(), 
            grad_level2.contiguous(), 
            grad_level3.contiguous(), 
            grad_input
        )
        
        return grad_input


class HaarQuadTransform(Function):
    """4-level cascade Haar transform with autograd support."""
    
    @staticmethod
    def forward(ctx, x):
        assert x.is_mps, "Input must be on MPS device"
        assert x.dim() == 4
        
        B, C, H, W = x.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        H4, W4 = (H + 3) // 4, (W + 3) // 4
        H8, W8 = (H + 7) // 8, (W + 7) // 8
        H16, W16 = (H + 15) // 16, (W + 15) // 16
        
        level1 = torch.empty((B, C, 4, H2, W2), dtype=x.dtype, device=x.device)
        level2 = torch.empty((B, C, 4, H4, W4), dtype=x.dtype, device=x.device)
        level3 = torch.empty((B, C, 4, H8, W8), dtype=x.dtype, device=x.device)
        level4 = torch.empty((B, C, 4, H16, W16), dtype=x.dtype, device=x.device)
        
        _get_module().haar2d_quad_forward(x.contiguous(), level1, level2, level3, level4)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = x.dtype
        
        return level1, level2, level3, level4
    
    @staticmethod
    def backward(ctx, grad_level1, grad_level2, grad_level3, grad_level4):
        H, W = ctx.shape_hw
        B, C = grad_level4.shape[:2]
        
        grad_input = torch.zeros((B, C, H, W), dtype=ctx.dtype, device=grad_level4.device)
        
        _get_module().haar2d_quad_backward(
            grad_level1.contiguous(), 
            grad_level2.contiguous(), 
            grad_level3.contiguous(), 
            grad_level4.contiguous(),
            grad_input
        )
        
        return grad_input


class HaarQuintTransform(Function):
    """5-level cascade Haar transform with autograd support."""
    
    @staticmethod
    def forward(ctx, x):
        assert x.is_mps, "Input must be on MPS device"
        assert x.dim() == 4
        
        B, C, H, W = x.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        H4, W4 = (H + 3) // 4, (W + 3) // 4
        H8, W8 = (H + 7) // 8, (W + 7) // 8
        H16, W16 = (H + 15) // 16, (W + 15) // 16
        H32, W32 = (H + 31) // 32, (W + 31) // 32
        
        level1 = torch.empty((B, C, 4, H2, W2), dtype=x.dtype, device=x.device)
        level2 = torch.empty((B, C, 4, H4, W4), dtype=x.dtype, device=x.device)
        level3 = torch.empty((B, C, 4, H8, W8), dtype=x.dtype, device=x.device)
        level4 = torch.empty((B, C, 4, H16, W16), dtype=x.dtype, device=x.device)
        level5 = torch.empty((B, C, 4, H32, W32), dtype=x.dtype, device=x.device)
        
        _get_module().haar2d_quint_forward(x.contiguous(), level1, level2, level3, level4, level5)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = x.dtype
        
        return level1, level2, level3, level4, level5
    
    @staticmethod
    def backward(ctx, grad_level1, grad_level2, grad_level3, grad_level4, grad_level5):
        H, W = ctx.shape_hw
        B, C = grad_level5.shape[:2]
        
        grad_input = torch.zeros((B, C, H, W), dtype=ctx.dtype, device=grad_level5.device)
        
        _get_module().haar2d_quint_backward(
            grad_level1.contiguous(), 
            grad_level2.contiguous(), 
            grad_level3.contiguous(), 
            grad_level4.contiguous(),
            grad_level5.contiguous(),
            grad_input
        )
        
        return grad_input


# =============================================================================
# Inverse Cascade Transforms
# =============================================================================

class InverseHaarDoubleTransform(Function):
    """2-level inverse cascade Haar transform."""
    
    @staticmethod
    def forward(ctx, level1, level2, output_size):
        B, C = level1.shape[:2]
        H, W = output_size
        
        output = torch.empty((B, C, H, W), dtype=level1.dtype, device=level1.device)
        
        _get_module().haar2d_double_backward(level1.contiguous(), level2.contiguous(), output)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = level1.dtype
        ctx.sizes = (level1.shape, level2.shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        H, W = ctx.shape_hw
        B, C = grad_output.shape[:2]
        size1, size2 = ctx.sizes
        
        grad_level1 = torch.empty(size1, dtype=ctx.dtype, device=grad_output.device)
        grad_level2 = torch.empty(size2, dtype=ctx.dtype, device=grad_output.device)
        
        _get_module().haar2d_double_forward(grad_output.contiguous(), grad_level1, grad_level2)
        
        return grad_level1, grad_level2, None


class InverseHaarTripleTransform(Function):
    """3-level inverse cascade Haar transform."""
    
    @staticmethod
    def forward(ctx, level1, level2, level3, output_size):
        B, C = level1.shape[:2]
        H, W = output_size
        
        output = torch.empty((B, C, H, W), dtype=level1.dtype, device=level1.device)
        
        _get_module().haar2d_triple_backward(
            level1.contiguous(), level2.contiguous(), level3.contiguous(), output)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = level1.dtype
        ctx.sizes = (level1.shape, level2.shape, level3.shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        size1, size2, size3 = ctx.sizes
        
        grad_level1 = torch.empty(size1, dtype=ctx.dtype, device=grad_output.device)
        grad_level2 = torch.empty(size2, dtype=ctx.dtype, device=grad_output.device)
        grad_level3 = torch.empty(size3, dtype=ctx.dtype, device=grad_output.device)
        
        _get_module().haar2d_triple_forward(
            grad_output.contiguous(), grad_level1, grad_level2, grad_level3)
        
        return grad_level1, grad_level2, grad_level3, None


class InverseHaarQuadTransform(Function):
    """4-level inverse cascade Haar transform."""
    
    @staticmethod
    def forward(ctx, level1, level2, level3, level4, output_size):
        B, C = level1.shape[:2]
        H, W = output_size
        
        output = torch.empty((B, C, H, W), dtype=level1.dtype, device=level1.device)
        
        _get_module().haar2d_quad_backward(
            level1.contiguous(), level2.contiguous(), 
            level3.contiguous(), level4.contiguous(), output)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = level1.dtype
        ctx.sizes = (level1.shape, level2.shape, level3.shape, level4.shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        size1, size2, size3, size4 = ctx.sizes
        
        grad_level1 = torch.empty(size1, dtype=ctx.dtype, device=grad_output.device)
        grad_level2 = torch.empty(size2, dtype=ctx.dtype, device=grad_output.device)
        grad_level3 = torch.empty(size3, dtype=ctx.dtype, device=grad_output.device)
        grad_level4 = torch.empty(size4, dtype=ctx.dtype, device=grad_output.device)
        
        _get_module().haar2d_quad_forward(
            grad_output.contiguous(), grad_level1, grad_level2, grad_level3, grad_level4)
        
        return grad_level1, grad_level2, grad_level3, grad_level4, None


class InverseHaarQuintTransform(Function):
    """5-level inverse cascade Haar transform."""
    
    @staticmethod
    def forward(ctx, level1, level2, level3, level4, level5, output_size):
        B, C = level1.shape[:2]
        H, W = output_size
        
        output = torch.empty((B, C, H, W), dtype=level1.dtype, device=level1.device)
        
        _get_module().haar2d_quint_backward(
            level1.contiguous(), level2.contiguous(), level3.contiguous(),
            level4.contiguous(), level5.contiguous(), output)
        
        ctx.shape_hw = (H, W)
        ctx.dtype = level1.dtype
        ctx.sizes = (level1.shape, level2.shape, level3.shape, level4.shape, level5.shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        size1, size2, size3, size4, size5 = ctx.sizes
        
        grad_level1 = torch.empty(size1, dtype=ctx.dtype, device=grad_output.device)
        grad_level2 = torch.empty(size2, dtype=ctx.dtype, device=grad_output.device)
        grad_level3 = torch.empty(size3, dtype=ctx.dtype, device=grad_output.device)
        grad_level4 = torch.empty(size4, dtype=ctx.dtype, device=grad_output.device)
        grad_level5 = torch.empty(size5, dtype=ctx.dtype, device=grad_output.device)
        
        _get_module().haar2d_quint_forward(
            grad_output.contiguous(), grad_level1, grad_level2, 
            grad_level3, grad_level4, grad_level5)
        
        return grad_level1, grad_level2, grad_level3, grad_level4, grad_level5, None


# =============================================================================
# Functional API
# =============================================================================

def haar2d(x: torch.Tensor) -> torch.Tensor:
    """Single-level 2D Haar wavelet transform."""
    return HaarTransform.apply(x)


def ihaar2d(x: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Inverse 2D Haar wavelet transform."""
    return InverseHaarTransform.apply(x, output_size)


def haar2d_double(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """2-level cascade Haar transform."""
    return HaarDoubleTransform.apply(x)


def haar2d_triple(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-level cascade Haar transform."""
    return HaarTripleTransform.apply(x)


def haar2d_quad(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """4-level cascade Haar transform."""
    return HaarQuadTransform.apply(x)


def haar2d_quint(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """5-level cascade Haar transform."""
    return HaarQuintTransform.apply(x)


def ihaar2d_double(level1, level2, output_size):
    """Inverse 2-level cascade Haar transform."""
    return InverseHaarDoubleTransform.apply(level1, level2, output_size)


def ihaar2d_triple(level1, level2, level3, output_size):
    """Inverse 3-level cascade Haar transform."""
    return InverseHaarTripleTransform.apply(level1, level2, level3, output_size)


def ihaar2d_quad(level1, level2, level3, level4, output_size):
    """Inverse 4-level cascade Haar transform."""
    return InverseHaarQuadTransform.apply(level1, level2, level3, level4, output_size)


def ihaar2d_quint(level1, level2, level3, level4, level5, output_size):
    """Inverse 5-level cascade Haar transform."""
    return InverseHaarQuintTransform.apply(level1, level2, level3, level4, level5, output_size)


# =============================================================================
# Wrapper Classes
# =============================================================================

class HaarMetal:
    """Wrapper class for single-level Haar wavelet transform."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HaarTransform.apply(x)
    
    def inverse(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        return InverseHaarTransform.apply(x, output_size)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class HaarDoubleMetal:
    """Wrapper class for 2-level cascade Haar transform."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return HaarDoubleTransform.apply(x)
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


class HaarTripleMetal:
    """Wrapper class for 3-level cascade Haar transform."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return HaarTripleTransform.apply(x)
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward(x)


class HaarQuadMetal:
    """Wrapper class for 4-level cascade Haar transform."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return HaarQuadTransform.apply(x)
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward(x)


class HaarQuintMetal:
    """Wrapper class for 5-level cascade Haar transform."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return HaarQuintTransform.apply(x)
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward(x)


# =============================================================================
# Scaled Depthwise Conv - Best for Training (Dynamic Weight Fusion)
# =============================================================================


class ScaledDepthwiseConvFunction(Function):
    """
    Fused depthwise conv + scale using dynamic weight fusion.
    
    Fuses scale into weight before conv: y = conv(x, scale * weight)
    This uses the backend (MPS/cuDNN) for both forward and backward,
    giving ~1.17x training speedup over separate conv + scale_mul.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                padding: int, groups: int) -> torch.Tensor:
        import torch.nn.functional as F
        
        # Fuse scale into weight: fused_weight = scale * weight
        fused_weight = scale.view(-1, 1, 1, 1) * weight
        output = F.conv2d(input, fused_weight, padding=padding, groups=groups)
        
        ctx.save_for_backward(input, weight, scale, fused_weight)
        ctx.padding = padding
        ctx.groups = groups
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.nn.functional as F
        
        input, weight, scale, fused_weight = ctx.saved_tensors
        padding = ctx.padding
        groups = ctx.groups
        
        # grad_input uses fused_weight (backend backward)
        grad_input = torch.nn.grad.conv2d_input(
            input.shape, fused_weight, grad_output, padding=padding, groups=groups
        )
        
        # grad_fused_weight (backend backward)
        grad_fused_weight = torch.nn.grad.conv2d_weight(
            input, weight.shape, grad_output, padding=padding, groups=groups
        )
        
        # Unfuse: grad_weight = grad_fused_weight (already scaled during forward)
        grad_weight = grad_fused_weight
        
        # grad_scale = sum(grad_fused_weight * weight) over spatial dims
        grad_scale = (grad_fused_weight * weight).sum(dim=(1, 2, 3), keepdim=True).view(1, -1, 1, 1)
        
        return grad_input, grad_weight, grad_scale, None, None


def scaled_depthwise_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    padding: int = 1
) -> torch.Tensor:
    """
    Scaled depthwise convolution: output = scale * depthwise_conv(input, weight)
    
    This is the RECOMMENDED function for training. It fuses scale into weights
    before the convolution, using the backend (MPS/cuDNN) for both forward and 
    backward passes. Provides ~1.17x training speedup over separate conv + scale_mul.
    
    Args:
        input: Input tensor (B, C, H, W), float32/float16, MPS or CUDA
        weight: Weight tensor (C, 1, K, K), depthwise conv weights
        scale: Scale tensor (1, C, 1, 1), per-channel scale
        padding: Padding size (typically kernel_size // 2)
        
    Returns:
        Output tensor (B, C, H, W): scale * conv(input, weight)
    """
    groups = input.size(1)  # Depthwise: groups = channels
    return ScaledDepthwiseConvFunction.apply(input, weight, scale, padding, groups)

