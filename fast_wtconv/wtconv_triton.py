"""
Pure Triton WTConv2d

Uses Triton kernels for all operations:
- Forward Pass: Fused "Haar -> Conv -> Scale" kernel for all decomposition levels
- Inverse Pass: Triton inverse Haar reconstruction (all levels)
- Base Conv: Triton scaled depthwise convolution

This provides a completely Triton-based implementation with no CUDA dependencies.
"""

import sys
from pathlib import Path
import math
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_wtconv.triton_haar.triton_haar import fused_haar_conv_scale
from fast_wtconv.triton_haar.triton_ihaar import (
    ihaar2d_fused, ihaar2d_double_fused, ihaar2d_triple_fused,
    ihaar2d_quad_fused, ihaar2d_quint_fused
)
from fast_wtconv.triton_haar.triton_depthwise import scaled_depthwise_conv


class WTConv2d(nn.Module):
    """
    Pure Triton WTConv2d using Triton kernels for all operations.
    
    Args:
        in_channels: Number of input/output channels
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
        
        # Base conv parameters
        self.base_weight = nn.Parameter(
            torch.Tensor(in_channels, 1, kernel_size, kernel_size)
        )
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        
        if bias:
            self.base_bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('base_bias', None)

        # Initialize base parameters (mimic nn.Conv2d)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.base_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.base_bias, -bound, bound)
        
        # Wavelet parameters
        self.wt_weights = nn.ParameterList()
        self.wt_scales = nn.ParameterList()
        
        for _ in range(wt_levels):
            # Init weights with small std dev for stability
            self.wt_weights.append(nn.Parameter(
                torch.randn(in_channels * 4, 1, kernel_size, kernel_size) * 0.02
            ))
            self.wt_scales.append(nn.Parameter(
                torch.ones(1, in_channels * 4, 1, 1) * 0.1
            ))
    
    @torch._dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Check if padding needed
        need_pad = (H & 1) or (W & 1)
        
        # Base Conv Path (compute before padding, uses original input)
        base_out = scaled_depthwise_conv(
            x, self.base_weight, self.base_scale, self.kernel_size // 2,
            bias=self.base_bias
        )
        
        # Pad if odd dimensions for wavelet transform
        if need_pad:
            x = F.pad(x, (0, W & 1, 0, H & 1))
            
        current_input = x
        level_outputs = [None] * self.wt_levels
        
        # Forward Pass: Fused Haar -> Conv -> Scale Loop
        for i, (weight, scale) in enumerate(zip(self.wt_weights, self.wt_scales)):
            # Shape for this level
            Hi, Wi = current_input.shape[2], current_input.shape[3]
            
            # If not the last level, also get raw LL for next level
            need_ll = (i < self.wt_levels - 1)
            
            if need_ll:
                # Output is already (B, C, 4, Hi/2, Wi/2) - no reshape needed
                out, ll_raw = fused_haar_conv_scale(
                    current_input, weight, scale, self.kernel_size,
                    return_ll=True
                )
                current_input = ll_raw
            else:
                out = fused_haar_conv_scale(
                    current_input, weight, scale, self.kernel_size,
                    return_ll=False
                )
            
            level_outputs[i] = out
        
        # ---------------------------------------------------------------------
        # Inverse Pass: Fused Triton Reconstruction + Addition
        # Eliminates separate add by fusing into iHaar kernel
        # ---------------------------------------------------------------------
        if self.wt_levels == 1:
            output = ihaar2d_fused(level_outputs[0], base_out, output_size=(H, W))
            
        elif self.wt_levels == 2:
            output = ihaar2d_double_fused(
                level_outputs[0], level_outputs[1], base_out, (H, W)
            )
            
        elif self.wt_levels == 3:
            output = ihaar2d_triple_fused(
                level_outputs[0], level_outputs[1], level_outputs[2], base_out, (H, W)
            )
            
        elif self.wt_levels == 4:
            output = ihaar2d_quad_fused(
                level_outputs[0], level_outputs[1], level_outputs[2], level_outputs[3],
                base_out, (H, W)
            )
            
        else:  # 5 levels
            output = ihaar2d_quint_fused(
                level_outputs[0], level_outputs[1], level_outputs[2], level_outputs[3], level_outputs[4],
                base_out, (H, W)
            )
        
        if self.do_stride is not None:
            output = self.do_stride(output)
        
        return output
