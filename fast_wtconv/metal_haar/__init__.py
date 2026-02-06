"""
Metal Haar Wavelet Transform Library

Unified library for 2D Haar wavelet transforms on Apple Metal.
Requires macOS with Metal-capable GPU and PyTorch MPS support.
"""

from .haar_metal import (
    # Functional API
    haar2d,
    ihaar2d,
    haar2d_double,
    haar2d_triple,
    haar2d_quad,
    haar2d_quint,
    ihaar2d_double,
    ihaar2d_triple,
    ihaar2d_quad,
    ihaar2d_quint,
    # Scaled depthwise conv
    scaled_depthwise_conv,
    ScaledDepthwiseConvFunction,
    # Wrapper classes
    HaarMetal,
    HaarDoubleMetal,
    HaarTripleMetal,
    HaarQuadMetal,
    HaarQuintMetal,
    # Autograd functions (for advanced use)
    HaarTransform,
    InverseHaarTransform,
    HaarDoubleTransform,
    HaarTripleTransform,
    HaarQuadTransform,
    HaarQuintTransform,
    InverseHaarDoubleTransform,
    InverseHaarTripleTransform,
    InverseHaarQuadTransform,
    InverseHaarQuintTransform,
)

__all__ = [
    'haar2d',
    'ihaar2d',
    'haar2d_double',
    'haar2d_triple',
    'haar2d_quad',
    'haar2d_quint',
    'ihaar2d_double',
    'ihaar2d_triple',
    'ihaar2d_quad',
    'ihaar2d_quint',
    'scaled_depthwise_conv',
    'ScaledDepthwiseConvFunction',
    'HaarMetal',
    'HaarDoubleMetal',
    'HaarTripleMetal',
    'HaarQuadMetal',
    'HaarQuintMetal',
    'HaarTransform',
    'InverseHaarTransform',
    'HaarDoubleTransform',
    'HaarTripleTransform',
    'HaarQuadTransform',
    'HaarQuintTransform',
    'InverseHaarDoubleTransform',
    'InverseHaarTripleTransform',
    'InverseHaarQuadTransform',
    'InverseHaarQuintTransform',
]


