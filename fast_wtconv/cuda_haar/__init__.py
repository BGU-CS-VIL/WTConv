"""
CUDA Haar Wavelet Transform Library

Unified library for 2D Haar wavelet transforms with multiple cascade levels.
"""

from .haar_cuda import (
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
    # Wrapper classes
    HaarCUDA,
    HaarDoubleCUDA,
    HaarTripleCUDA,
    HaarQuadCUDA,
    HaarQuintCUDA,
    # Autograd functions (for advanced use)
    HaarTransform,
    InverseHaarTransform,
    HaarDoubleTransform,
    HaarTripleTransform,
    HaarQuadTransform,
    HaarQuintTransform,
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
    'HaarCUDA',
    'HaarDoubleCUDA',
    'HaarTripleCUDA',
    'HaarQuadCUDA',
    'HaarQuintCUDA',
    'HaarTransform',
    'InverseHaarTransform',
    'HaarDoubleTransform',
    'HaarTripleTransform',
    'HaarQuadTransform',
    'HaarQuintTransform',
]
