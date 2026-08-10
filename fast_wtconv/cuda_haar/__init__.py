"""
Fused CUDA Haar kernels for WTConv.

The Haar transform is fused into the depthwise convolution weights, so wavelet
coefficients never touch memory.
"""

from .haar_cuda import (
    # Largest kernel size this build is compiled for (see HAAR_MAX_K), the hard
    # ceiling of the kernels themselves, and the check that reports the difference
    MAX_KERNEL_SIZE,
    HARD_MAX_KERNEL_SIZE,
    check_kernel_size,
    # Whole wavelet branch (fused Haar+conv+scale per level, fused inverse+add)
    wavelet_branch,
    # Fused Haar -> conv -> scale
    fused_haar_conv_scale,
    compute_scaled_weight,
    # Inverse cascade with the final add fused in
    ihaar2d_fused,
    ihaar2d_double_fused,
    ihaar2d_triple_fused,
    ihaar2d_quad_fused,
    ihaar2d_quint_fused,
    # Plain inverse cascade
    ihaar2d,
    ihaar2d_double,
    ihaar2d_triple,
    ihaar2d_quad,
    ihaar2d_quint,
    # Forward Haar (utility)
    haar2d,
    # Base-conv path
    scaled_depthwise_conv,
    # Autograd functions / raw runners (advanced use)
    FusedHaarConvScaleFunction,
    WaveletBranchFunction,
    IHaarCascadeFn,
    run_ihaar_cascade,
    run_haar_cascade,
)

__all__ = [
    'MAX_KERNEL_SIZE',
    'HARD_MAX_KERNEL_SIZE',
    'check_kernel_size',
    'wavelet_branch',
    'fused_haar_conv_scale',
    'compute_scaled_weight',
    'ihaar2d_fused',
    'ihaar2d_double_fused',
    'ihaar2d_triple_fused',
    'ihaar2d_quad_fused',
    'ihaar2d_quint_fused',
    'ihaar2d',
    'ihaar2d_double',
    'ihaar2d_triple',
    'ihaar2d_quad',
    'ihaar2d_quint',
    'haar2d',
    'scaled_depthwise_conv',
    'FusedHaarConvScaleFunction',
    'WaveletBranchFunction',
    'IHaarCascadeFn',
    'run_ihaar_cascade',
    'run_haar_cascade',
]
