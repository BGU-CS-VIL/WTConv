"""
WTConv2d on fused CUDA kernels.

The whole wavelet branch is one autograd node (cuda_haar.wavelet_branch):

    levels 1..L    ONE kernel per level does Haar + depthwise conv + scale, with
                   the scale folded into the conv weights, so the wavelet
                   coefficients are formed on chip and only the filtered result
                   (plus the raw LL the next level needs) reaches memory.
    reconstruction ONE kernel walks the whole inverse cascade in registers and
                   folds the base-conv add into its final store, so no
                   intermediate low-pass or reconstruction tensor is written.

The base convolution keeps its own scaled depthwise conv (cuDNN, with the scale
folded into weight and bias).

Numerics follow the reference WTConv implementation of Finder et al. (2024)
exactly, including the per-level zero padding of odd spatial sizes and the crop
on reconstruction.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

_cuda_haar = None


def _get_haar_module():
    """Import the fused CUDA kernels, deferred so they compile on first use."""
    global _cuda_haar
    if _cuda_haar is None:
        from fast_wtconv.cuda_haar import haar_cuda as _cuda_haar_import
        _cuda_haar = _cuda_haar_import
    return _cuda_haar


class WTConv2d(nn.Module):
    """
    Args:
        in_channels: Number of input/output channels (must be equal)
        out_channels: Must equal in_channels
        kernel_size: Convolution kernel size, odd and <= 29 (default: 5). Sizes
            above 7 need the kernels rebuilt with HAAR_MAX_K set (see README).
        stride: Output stride, applied as average pooling (default: 1)
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

        self._haar = _get_haar_module()

        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        assert wt_levels in [1, 2, 3, 4, 5], "wt_levels must be 1-5"
        # Fails here, before anything is compiled, if this build has no such K.
        self._haar.check_kernel_size(kernel_size)

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.kernel_size = kernel_size
        self.stride = stride

        # Stride support via average pooling (matches the reference implementation)
        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

        # Base conv parameters (weights are plain tensors: they get fused with
        # base_scale before the convolution)
        self.base_weight = nn.Parameter(
            torch.empty(in_channels, 1, kernel_size, kernel_size)
        )
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        if bias:
            self.base_bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('base_bias', None)

        # Wavelet level parameters: channel c*4+s holds subband s of channel c
        self.wt_weights = nn.ParameterList()
        self.wt_scales = nn.ParameterList()
        for _ in range(wt_levels):
            self.wt_weights.append(nn.Parameter(
                torch.empty(in_channels * 4, 1, kernel_size, kernel_size)
            ))
            self.wt_scales.append(nn.Parameter(
                torch.ones(1, in_channels * 4, 1, 1) * 0.1
            ))

        self.reset_parameters()

    def reset_parameters(self):
        """Match nn.Conv2d's default initialisation, as in the reference model."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.base_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.base_bias, -bound, bound)
        for w in self.wt_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        haar = self._haar
        K = self.kernel_size

        # Base conv at full resolution, on the unpadded input
        base_out = haar.scaled_depthwise_conv(
            x, self.base_weight, self.base_scale, K // 2, bias=self.base_bias
        )

        # Wavelet branch: all levels, the inverse cascade and the base add.
        # Odd sizes are zero padded per level, as the reference does.
        output = haar.wavelet_branch(
            x, base_out, list(self.wt_weights), list(self.wt_scales), K
        )

        if self.do_stride is not None:
            output = self.do_stride(output)
        return output
