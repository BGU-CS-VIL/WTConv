"""
WTConv2d on the Metal (MPS) Haar kernels.

Per level the Haar coefficients are produced by a cascade kernel, filtered by a
scaled depthwise convolution, and the levels are reconstructed by an inverse
cascade kernel. Apple Silicon only; see wtconv_cuda.py for the CUDA backend and
wtconv_triton.py for the Triton one.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

_metal_haar = None


def _get_haar_module():
    """Import the Metal kernels, deferred so they compile on first use."""
    global _metal_haar
    if _metal_haar is None:
        from fast_wtconv.metal_haar import haar_metal as _metal_haar_import
        _metal_haar = _metal_haar_import
    return _metal_haar


class WTConv2d(nn.Module):
    """
    WTConv2d with Optimized Cascade Kernels (Metal / MPS).
    Args:
        in_channels: Number of input/output channels (must be equal)
        out_channels: Must equal in_channels
        kernel_size: Convolution kernel size (default: 5)
        wt_levels: Number of wavelet decomposition levels (1-5)
        bias: Include bias in base convolution (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        wt_levels: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        assert wt_levels in [1, 2, 3, 4, 5], "wt_levels must be 1-5"

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.kernel_size = kernel_size
        self.stride = stride

        # Stride support via average pooling (matches original implementation)
        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

        self._haar = _get_haar_module()

        # Base conv at full resolution
        self.base_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding='same', groups=in_channels, bias=bias
        )
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # Wavelet level convolutions
        self.wavelet_convs = nn.ModuleList()
        self.wavelet_scales = nn.ParameterList()
        for _ in range(wt_levels):
            conv = nn.Conv2d(
                in_channels * 4, in_channels * 4, kernel_size,
                padding='same', groups=in_channels * 4, bias=False
            )
            self.wavelet_convs.append(conv)
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        padding = self.kernel_size // 2
        haar = self._haar

        # Pad if odd dimensions
        if (H & 1) or (W & 1):
            x = F.pad(x, (0, W & 1, 0, H & 1))

        # Wavelet path
        if self.wt_levels == 1:
            level1 = haar.haar2d(x)
            level1_conv = self._apply_conv(level1, 0, padding, haar)
            output_wt = haar.ihaar2d(level1_conv, output_size=(H, W))

        elif self.wt_levels == 2:
            level1, level2 = haar.haar2d_double(x)
            level1_conv = self._apply_conv(level1, 0, padding, haar)
            level2_conv = self._apply_conv(level2, 1, padding, haar)
            output_wt = haar.ihaar2d_double(level1_conv, level2_conv, (H, W))

        elif self.wt_levels == 3:
            level1, level2, level3 = haar.haar2d_triple(x)
            level1_conv = self._apply_conv(level1, 0, padding, haar)
            level2_conv = self._apply_conv(level2, 1, padding, haar)
            level3_conv = self._apply_conv(level3, 2, padding, haar)
            output_wt = haar.ihaar2d_triple(level1_conv, level2_conv, level3_conv, (H, W))

        elif self.wt_levels == 4:
            level1, level2, level3, level4 = haar.haar2d_quad(x)
            level1_conv = self._apply_conv(level1, 0, padding, haar)
            level2_conv = self._apply_conv(level2, 1, padding, haar)
            level3_conv = self._apply_conv(level3, 2, padding, haar)
            level4_conv = self._apply_conv(level4, 3, padding, haar)
            output_wt = haar.ihaar2d_quad(level1_conv, level2_conv, level3_conv, level4_conv, (H, W))

        else:  # wt_levels == 5
            level1, level2, level3, level4, level5 = haar.haar2d_quint(x)
            level1_conv = self._apply_conv(level1, 0, padding, haar)
            level2_conv = self._apply_conv(level2, 1, padding, haar)
            level3_conv = self._apply_conv(level3, 2, padding, haar)
            level4_conv = self._apply_conv(level4, 3, padding, haar)
            level5_conv = self._apply_conv(level5, 4, padding, haar)
            output_wt = haar.ihaar2d_quint(level1_conv, level2_conv, level3_conv, level4_conv, level5_conv, (H, W))

        # Base conv
        base_out = haar.scaled_depthwise_conv(x[:, :, :H, :W], self.base_conv.weight, self.base_scale, padding)
        if self.base_conv.bias is not None:
            base_out = base_out + self.base_conv.bias.view(1, -1, 1, 1)

        output = base_out + output_wt

        if self.do_stride is not None:
            output = self.do_stride(output)

        return output

    def _apply_conv(self, coeffs: torch.Tensor, level: int, padding: int, haar) -> torch.Tensor:
        B, C, _, h, w = coeffs.shape
        flat = coeffs.view(B, C * 4, h, w)
        out = haar.scaled_depthwise_conv(flat, self.wavelet_convs[level].weight, self.wavelet_scales[level], padding)
        return out.view(B, C, 4, h, w)
