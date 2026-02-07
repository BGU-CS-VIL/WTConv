"""
Triton Haar Wavelet Transform Module

Provides fused Haar → Conv → Scale kernels for Hybrid WTConv.
"""

from .triton_haar import (
    fused_haar_conv_scale,
    compute_scaled_weight,
)

__all__ = [
    'fused_haar_conv_scale',
    'compute_scaled_weight',
]
