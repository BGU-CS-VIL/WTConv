"""
Triton Fused Haar Wavelet + Depthwise Convolution Scale Kernels

Contains the core Triton kernel for the Hybrid WTConv approach:
- fused_haar_conv_scale: Run fused Haar -> Conv -> Scale using an optimized
  kernel that exploits Haar symmetry for 4x reduced kernel memory.

Used by `wtconv_hybrid.py`.
"""

import torch
import torch._dynamo
import triton
import triton.language as tl
from typing import Tuple


def compute_scaled_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int = 3
) -> torch.Tensor:
    """
    Precompute scaled weights for the optimized kernel.
    
    Instead of a (C, 4, K2, K2) effective kernel, we compute (C, 4, K, K)
    weights that operate on Haar partial sums directly.
    
    Args:
        weight: (C*4, 1, K, K) depthwise conv weights
        scale: (1, C*4, 1, 1) or (C*4,) scales
        kernel_size: Conv kernel size K
        
    Returns:
        scaled_weight: (C, 4, K, K) - scaled weights per channel per subband
    """
    C4 = weight.shape[0]
    C = C4 // 4
    K = kernel_size
    
    # Apply scale to weights: (C*4, 1, K, K)
    scale_flat = scale.view(C4)
    scaled_weight = weight.view(C4, K, K) * scale_flat.view(C4, 1, 1)
    
    # Reshape to (C, 4, K, K) - 4 subbands per channel
    return scaled_weight.view(C, 4, K, K).contiguous()


@triton.autotune(
    configs=[
        # Smaller blocks for smaller inputs
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=3),
        # Medium blocks - good balance
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=2),
        # Larger blocks for larger inputs
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
        # Maximum occupancy configs
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def _fused_haar_conv_scale_kernel(
    # Pointers
    input_ptr,
    scaled_weight_ptr,  # (C, 4, K*K) - much smaller than effective kernel!
    output_ptr,
    ll_output_ptr,
    # Dimensions
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_ib: tl.constexpr,
    stride_ic: tl.constexpr,
    stride_ih: tl.constexpr,
    stride_iw: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oc: tl.constexpr,
    stride_os: tl.constexpr,  # subband stride (dim 2 in B,C,4,H,W)
    stride_oh: tl.constexpr,
    stride_ow: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lc: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lw: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Haar -> Conv -> Scale kernel exploiting Haar symmetry.
    
    Key insight: For each 2x2 input block at (2*kh, 2*kw), the Haar transform
    produces partial sums:
        S  = x00 + x01 + x10 + x11  (sum, used by LL)
        Dh = x00 - x01 + x10 - x11  (horizontal diff, used by LH)
        Dv = x00 + x01 - x10 - x11  (vertical diff, used by HL)
        Dd = x00 - x01 - x10 + x11  (diagonal diff, used by HH)
    
    Instead of loading 4 kernel weights per position, we:
    1. Load 4 inputs per 2x2 block
    2. Compute 4 partial sums (same cost as before, just reordered)
    3. Load 4 kernel weights (one per subband for this K position)
    4. FMA: out_s += w_s * partial_s
    
    This reduces kernel memory traffic by 4x (K*K weights vs K2*K2*4).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode position
    w2 = offs % W2
    tmp = offs // W2
    h2 = tmp % H2
    tmp = tmp // H2
    c = tmp % C
    b = tmp // C
    
    # Top-left of output's corresponding 2x2 input block
    ih_base = h2 * 2
    iw_base = w2 * 2
    
    # Input and weight bases
    in_base = input_ptr + b * stride_ib + c * stride_ic
    wt_base = scaled_weight_ptr + c * 4 * K * K  # (C, 4, K, K) flattened
    
    # Accumulators
    ll_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    lh_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    hl_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    hh_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Raw LL accumulator (if needed)
    ll_raw = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over K x K kernel positions (not K2 x K2!)
    # Each position corresponds to a 2x2 input block
    for kh in tl.static_range(K):
        for kw in tl.static_range(K):
            # 2x2 block top-left in input
            # Center the kernel: offset by -(K-1)//2 in Haar-downsampled space
            # which is -2*((K-1)//2) = -(K-1) in original space (for K=3, this is -2)
            # But we need to match the original conv padding behavior
            ih0 = ih_base + 2 * kh - (K - 1)
            iw0 = iw_base + 2 * kw - (K - 1)
            
            # Load 4 input values for the 2x2 block
            # (ih0, iw0), (ih0, iw0+1), (ih0+1, iw0), (ih0+1, iw0+1)
            
            valid00 = mask & (ih0 >= 0) & (ih0 < H) & (iw0 >= 0) & (iw0 < W)
            valid01 = mask & (ih0 >= 0) & (ih0 < H) & (iw0 + 1 >= 0) & (iw0 + 1 < W)
            valid10 = mask & (ih0 + 1 >= 0) & (ih0 + 1 < H) & (iw0 >= 0) & (iw0 < W)
            valid11 = mask & (ih0 + 1 >= 0) & (ih0 + 1 < H) & (iw0 + 1 >= 0) & (iw0 + 1 < W)
            
            x00 = tl.load(in_base + ih0 * stride_ih + iw0 * stride_iw, mask=valid00, other=0.0)
            x01 = tl.load(in_base + ih0 * stride_ih + (iw0 + 1) * stride_iw, mask=valid01, other=0.0)
            x10 = tl.load(in_base + (ih0 + 1) * stride_ih + iw0 * stride_iw, mask=valid10, other=0.0)
            x11 = tl.load(in_base + (ih0 + 1) * stride_ih + (iw0 + 1) * stride_iw, mask=valid11, other=0.0)
            
            # Compute Haar partial sums - these are the 4 subband contributions
            # Matching CUDA haar_step: ll = 0.5*(a+b+c+d), lh = 0.5*(a+b-c-d), 
            #                          hl = 0.5*(a-b+c-d), hh = 0.5*(a-b-c+d)
            # Where a=x00, b=x01, c=x10, d=x11
            S  = (x00 + x01 + x10 + x11) * 0.5  # LL: average
            Dh = (x00 + x01 - x10 - x11) * 0.5  # LH: horizontal low, vertical high
            Dv = (x00 - x01 + x10 - x11) * 0.5  # HL: horizontal high, vertical low
            Dd = (x00 - x01 - x10 + x11) * 0.5  # HH: diagonal difference
            
            # Load weights for this kernel position - only 4 loads total!
            k_idx = kh * K + kw
            w_ll = tl.load(wt_base + 0 * K * K + k_idx)
            w_lh = tl.load(wt_base + 1 * K * K + k_idx)
            w_hl = tl.load(wt_base + 2 * K * K + k_idx)
            w_hh = tl.load(wt_base + 3 * K * K + k_idx)
            
            # Accumulate: each subband output is conv of its partial sums
            ll_out += w_ll * S
            lh_out += w_lh * Dh
            hl_out += w_hl * Dv
            hh_out += w_hh * Dd
            
            # Accumulate raw LL at center position
            # Center position is at kh = (K-1)//2, kw = (K-1)//2 for K=3 this is (1,1)
            # But we can compute it for any K by checking
            if kh == (K - 1) // 2:
                if kw == (K - 1) // 2:
                    ll_raw = S  # S = (x00+x01+x10+x11)*0.5 = LL
    
    # Store outputs - using (B, C, 4, H2, W2) layout
    out_base = output_ptr + b * stride_ob + c * stride_oc
    tl.store(out_base + 0 * stride_os + h2 * stride_oh + w2 * stride_ow, ll_out, mask=mask)
    tl.store(out_base + 1 * stride_os + h2 * stride_oh + w2 * stride_ow, lh_out, mask=mask)
    tl.store(out_base + 2 * stride_os + h2 * stride_oh + w2 * stride_ow, hl_out, mask=mask)
    tl.store(out_base + 3 * stride_os + h2 * stride_oh + w2 * stride_ow, hh_out, mask=mask)
    
    if ll_output_ptr is not None:
        ll_base = ll_output_ptr + b * stride_lb + c * stride_lc
        tl.store(ll_base + h2 * stride_lh + w2 * stride_lw, ll_raw, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def _compute_haar_coeffs_kernel(
    input_ptr,       # (B, C, H, W)
    output_ptr,      # (B, C*4, H2, W2)
    B, C, H, W, H2, W2, N,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode output position (same as output grid of one subband: B * C * H2 * W2)
    w2 = offs % W2
    tmp = offs // W2
    h2 = tmp % H2
    tmp = tmp // H2
    c = tmp % C
    b = tmp // C
    
    # Input 2x2 block top-left
    ih = h2 * 2
    iw = w2 * 2
    
    in_base = input_ptr + b * stride_ib + c * stride_ic
    
    # Load 2x2 block
    x00_ptr = in_base + ih * stride_ih + iw * stride_iw
    x01_ptr = in_base + ih * stride_ih + (iw + 1) * stride_iw
    x10_ptr = in_base + (ih + 1) * stride_ih + iw * stride_iw
    x11_ptr = in_base + (ih + 1) * stride_ih + (iw + 1) * stride_iw
    
    # Safe load with masking
    x00 = tl.load(x00_ptr, mask=mask, other=0.0)
    x01 = tl.load(x01_ptr, mask=mask, other=0.0)
    x10 = tl.load(x10_ptr, mask=mask, other=0.0)
    x11 = tl.load(x11_ptr, mask=mask, other=0.0)
    
    # Compute Haar subbands
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    
    # Output pointers
    # Output shape is (B, C*4, H2, W2)
    # The 4 subbands for channel c are at c*4, c*4+1, c*4+2, c*4+3
    out_base = output_ptr + b * stride_ob + (c * 4) * stride_oc + h2 * stride_oh + w2 * stride_ow
    
    tl.store(out_base + 0 * stride_oc, ll, mask=mask)
    tl.store(out_base + 1 * stride_oc, lh, mask=mask)
    tl.store(out_base + 2 * stride_oc, hl, mask=mask)
    tl.store(out_base + 3 * stride_oc, hh, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def _fused_haar_conv_scale_backward_kernel(
    # Pointers
    grad_output_ptr,  # (B, C*4, H2, W2)
    grad_ll_ptr,      # (B, C, H2, W2)
    scaled_weight_ptr,  # (C, 4, K*K) - flattened
    grad_input_ptr,  # (B, C, H, W)
    # Dimensions
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    N: tl.constexpr,  # B * C * H2 * W2 (same as forward!)
    K: tl.constexpr,
    # Strides for grad_output (B, C, 4, H2, W2)
    stride_gob: tl.constexpr,
    stride_goc: tl.constexpr,
    stride_gos: tl.constexpr,  # subband stride
    stride_goh: tl.constexpr,
    stride_gow: tl.constexpr,
    # Strides for grad_ll
    stride_glb: tl.constexpr,
    stride_glc: tl.constexpr,
    stride_glh: tl.constexpr,
    stride_glw: tl.constexpr,
    # Strides for grad_input
    stride_gib: tl.constexpr,
    stride_gic: tl.constexpr,
    stride_gih: tl.constexpr,
    stride_giw: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LL: tl.constexpr,
):
    """
    Optimized backward kernel - same structure as forward!
    
    Iterates over output positions (H2 x W2), computing gradients for a 2x2 input block.
    Each thread writes to a UNIQUE 2x2 block, so no atomics needed.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode position (same as forward!)
    w2 = offs % W2
    tmp = offs // W2
    h2 = tmp % H2
    tmp = tmp // H2
    c = tmp % C
    b = tmp // C
    
    # 2x2 input block top-left position
    ih_base = h2 * 2
    iw_base = w2 * 2
    
    # Pointers - using (B, C, 4, H2, W2) layout
    grad_out_base = grad_output_ptr + b * stride_gob + c * stride_goc
    wt_base = scaled_weight_ptr + c * 4 * K * K
    grad_in_base = grad_input_ptr + b * stride_gib + c * stride_gic
    
    # Accumulators for 4 pixels in the 2x2 input block
    # Accumulators for 4 pixels in the 2x2 input block
    grad_00 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    grad_01 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    grad_10 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    grad_11 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    if HAS_LL:
        gll_ptr = grad_ll_ptr + b * stride_glb + c * stride_glc + h2 * stride_glh + w2 * stride_glw
        val = tl.load(gll_ptr, mask=mask, other=0.0) * 0.5
        grad_00 += val
        grad_01 += val
        grad_10 += val
        grad_11 += val
    
    # Center offset for kernel
    center = (K - 1) // 2
    
    # Iterate over K x K kernel positions
    for kh in tl.static_range(K):
        for kw in tl.static_range(K):
            # Which output position read from our input block using kernel (kh, kw)?
            out_h2 = h2 - kh + center
            out_w2 = w2 - kw + center
            
            valid = mask & (out_h2 >= 0) & (out_h2 < H2) & (out_w2 >= 0) & (out_w2 < W2)
            
            # Load gradients from this output position - using (B, C, 4, H2, W2) layout
            idx_base = out_h2 * stride_goh + out_w2 * stride_gow
            grad_ll = tl.load(
                grad_out_base + 0 * stride_gos + idx_base,
                mask=valid, other=0.0
            )
            grad_lh = tl.load(
                grad_out_base + 1 * stride_gos + idx_base,
                mask=valid, other=0.0
            )
            grad_hl = tl.load(
                grad_out_base + 2 * stride_gos + idx_base,
                mask=valid, other=0.0
            )
            grad_hh = tl.load(
                grad_out_base + 3 * stride_gos + idx_base,
                mask=valid, other=0.0
            )
            
            # Load kernel weights for this position
            k_idx = kh * K + kw
            w_ll = tl.load(wt_base + 0 * K * K + k_idx)
            w_lh = tl.load(wt_base + 1 * K * K + k_idx)
            w_hl = tl.load(wt_base + 2 * K * K + k_idx)
            w_hh = tl.load(wt_base + 3 * K * K + k_idx)
            
            # Weighted gradients per subband
            wg_ll = w_ll * grad_ll
            wg_lh = w_lh * grad_lh
            wg_hl = w_hl * grad_hl
            wg_hh = w_hh * grad_hh
            
            # Apply inverse Haar to distribute to 4 input positions
            # Forward Haar: ll=0.5*(x00+x01+x10+x11), etc.
            # Transpose (backward): 
            grad_00 += 0.5 * (wg_ll + wg_lh + wg_hl + wg_hh)
            grad_01 += 0.5 * (wg_ll + wg_lh - wg_hl - wg_hh)
            grad_10 += 0.5 * (wg_ll - wg_lh + wg_hl - wg_hh)
            grad_11 += 0.5 * (wg_ll - wg_lh - wg_hl + wg_hh)
    
    # Store gradients to the 4 input positions
    valid00 = mask & (ih_base < H) & (iw_base < W)
    valid01 = mask & (ih_base < H) & (iw_base + 1 < W)
    valid10 = mask & (ih_base + 1 < H) & (iw_base < W)
    valid11 = mask & (ih_base + 1 < H) & (iw_base + 1 < W)
    
    tl.store(grad_in_base + ih_base * stride_gih + iw_base * stride_giw, grad_00, mask=valid00)
    tl.store(grad_in_base + ih_base * stride_gih + (iw_base + 1) * stride_giw, grad_01, mask=valid01)
    tl.store(grad_in_base + (ih_base + 1) * stride_gih + iw_base * stride_giw, grad_10, mask=valid10)
    tl.store(grad_in_base + (ih_base + 1) * stride_gih + (iw_base + 1) * stride_giw, grad_11, mask=valid11)


def _compute_grad_weight_scale(
    haar_coeffs: torch.Tensor,  # Pre-computed Haar coefficients (B, C*4, H2, W2)
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients for weight and scale.
    
    Uses pre-computed Haar coefficients from forward pass to avoid recomputation.
    """
    K = kernel_size
    C4 = haar_coeffs.shape[1]
    
    scale_flat = scale.view(C4)
    padding = (K - 1) // 2
    
    # Compute grad_fused_weight using conv2d_weight
    grad_fused_weight = torch.nn.grad.conv2d_weight(
        haar_coeffs, weight.shape, grad_output, padding=padding, groups=C4
    )
    
    # grad_weight = grad_fused_weight * scale (chain rule)
    grad_weight = grad_fused_weight * scale_flat.view(C4, 1, 1, 1)
    
    # grad_scale = sum(grad_fused_weight * weight) over spatial dims
    grad_scale = (grad_fused_weight * weight).sum(dim=(1, 2, 3)).view(1, C4, 1, 1)
    
    return grad_weight, grad_scale


def _compute_haar_coeffs(x: torch.Tensor) -> torch.Tensor:
    """Compute Haar coefficients from input tensor - used for weight gradient."""
    B, C, H, W = x.shape
    H2, W2 = H // 2, W // 2
    
    output = torch.empty(B, C * 4, H2, W2, device=x.device, dtype=x.dtype)
    
    N = B * C * H2 * W2
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    _compute_haar_coeffs_kernel[grid](
        x, output,
        B, C, H, W, H2, W2, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


@torch._dynamo.allow_in_graph
class FusedHaarConvScaleFunction(torch.autograd.Function):
    """
    Autograd function for fused Haar -> Conv -> Scale.
    
    Forward: Applies Haar transform, depthwise conv with scaled weights.
    Backward: Computes gradients for input, weight, and scale.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        kernel_size: int,
        return_ll: bool,
    ):
        assert x.is_cuda, "Input must be on CUDA"
        assert x.dim() == 4, "Input must be (B, C, H, W)"
        
        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2
        K = kernel_size
        
        # Allocate output directly in (B, C, 4, H2, W2) layout
        output = torch.empty(B, C, 4, H2, W2, device=x.device, dtype=x.dtype)
        
        ll_output = None
        lp_ptr = None
        stride_lb, stride_lc, stride_lh, stride_lw = 0, 0, 0, 0
        
        if return_ll:
            ll_output = torch.empty(B, C, H2, W2, device=x.device, dtype=x.dtype)
            lp_ptr = ll_output
            stride_lb, stride_lc, stride_lh, stride_lw = ll_output.stride()
        
        N = B * C * H2 * W2
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        
        # Compute scaled weights (C, 4, K, K)
        scaled_weight = compute_scaled_weight(weight, scale, K)
        sw_flat = scaled_weight.view(-1).contiguous()
        
        _fused_haar_conv_scale_kernel[grid](
            x, sw_flat, output, lp_ptr,
            B, C, H, W, H2, W2, N, K,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            stride_lb, stride_lc, stride_lh, stride_lw,
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, scale, scaled_weight)
        ctx.kernel_size = K
        ctx.return_ll = return_ll
        
        if return_ll:
            return output, ll_output
        return output
    
    @staticmethod
    def backward(ctx, grad_output, grad_ll=None):
        x, weight, scale, scaled_weight = ctx.saved_tensors
        K = ctx.kernel_size
        
        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2
        
        grad_input = None
        grad_weight = None
        grad_scale = None
        
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        needs_scale_grad = ctx.needs_input_grad[2]
        
        # Compute grad_input using optimized Triton backward kernel
        if needs_input_grad:
            grad_input = torch.empty(B, C, H, W, device=x.device, dtype=x.dtype)
            
            # Same grid size as forward (iterate over output positions)
            N = B * C * H2 * W2
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
            
            sw_flat = scaled_weight.view(-1).contiguous()
            
            # Prepare grad_ll args
            grad_ll_ptr = grad_output # dummy
            stride_glb, stride_glc, stride_glh, stride_glw = 0, 0, 0, 0
            has_ll = False
            
            if grad_ll is not None:
                has_ll = True
                grad_ll = grad_ll.contiguous()
                grad_ll_ptr = grad_ll
                stride_glb, stride_glc, stride_glh, stride_glw = grad_ll.stride()

            _fused_haar_conv_scale_backward_kernel[grid](
                grad_output.contiguous(), grad_ll_ptr, sw_flat, grad_input,
                B, C, H, W, H2, W2, N, K,
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3), grad_output.stride(4),
                stride_glb, stride_glc, stride_glh, stride_glw,
                grad_input.stride(0), grad_input.stride(1), grad_input.stride(2), grad_input.stride(3),
                HAS_LL=has_ll
            )
        
        # Compute grad_weight and grad_scale
        if needs_weight_grad or needs_scale_grad:
            haar_coeffs = _compute_haar_coeffs(x)
            # Reshape grad_output from (B, C, 4, H2, W2) to (B, C*4, H2, W2) for conv2d_weight
            grad_output_flat = grad_output.view(B, C * 4, H2, W2)
            grad_weight, grad_scale = _compute_grad_weight_scale(
                haar_coeffs, grad_output_flat, weight, scale, K
            )
        
        return grad_input, grad_weight, grad_scale, None, None


def fused_haar_conv_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int = 3,
    return_ll: bool = False,
) -> torch.Tensor:
    """
    Fused Haar Transform -> Depthwise Conv -> Scale.
    
    Uses an optimized kernel that exploits Haar symmetry for 4x reduced
    kernel memory traffic compared to the naive effective kernel approach.
    
    This function supports automatic differentiation through FusedHaarConvScaleFunction.
    
    Args:
        x: Input tensor (B, C, H, W)
        weight: (C*4, 1, K, K) depthwise conv weights
        scale: (1, C*4, 1, 1) or (C*4,) scales
        kernel_size: K
        return_ll: If True, returns (coeffs, ll_raw) where ll_raw is (B, C, H/2, W/2)
                   containing the raw LL subband (AvgPool) for the next level.
    
    Returns:
        coeffs: (B, C*4, H/2, W/2)
        ll_raw: (B, C, H/2, W/2) (only if return_ll=True)
    """
    return FusedHaarConvScaleFunction.apply(x, weight, scale, kernel_size, return_ll)
