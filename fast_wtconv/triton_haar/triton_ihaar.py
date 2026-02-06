"""
Triton Inverse Haar Transform Implementation

Provides Triton kernels for single and multi-level inverse Haar transforms.
Replicates the functionality of `cuda_haar` but using Triton.

Optimizations:
- Selective reconstruction math for intermediate levels
- FMA-friendly computations
- 4 outputs per thread (better compute/load ratio than 1 output per thread)
"""

import torch
import torch._dynamo
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _ihaar_step(ll, lh, hl, hh):
    """
    Inverse Haar step: Reconstruct 2x2 block from 4 subbands.
    Uses FMA-friendly computation pattern.
    """
    # Pre-compute partial sums for better instruction scheduling
    ll_plus_lh = ll + lh
    ll_minus_lh = ll - lh
    hl_plus_hh = hl + hh
    hl_minus_hh = hl - hh
    
    # Final reconstruction with 0.5 scaling
    a = 0.5 * (ll_plus_lh + hl_plus_hh)
    b = 0.5 * (ll_plus_lh - hl_plus_hh)
    c = 0.5 * (ll_minus_lh + hl_minus_hh)
    d = 0.5 * (ll_minus_lh - hl_minus_hh)
    return a, b, c, d


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _ihaar_cascade_kernel(
    # Input pointers for up to 5 levels
    l1_ptr, l2_ptr, l3_ptr, l4_ptr, l5_ptr,
    output_ptr,
    # Optional add tensor (fused addition)
    add_ptr,
    # Dimensions for final output
    H: tl.constexpr, 
    W: tl.constexpr,
    # Dimensions for each level
    H1: tl.constexpr, W1: tl.constexpr,
    H2: tl.constexpr, W2: tl.constexpr,
    H3: tl.constexpr, W3: tl.constexpr,
    H4: tl.constexpr, W4: tl.constexpr,
    H5: tl.constexpr, W5: tl.constexpr,
    # Strides
    stride_l1_plane: tl.constexpr, stride_l1_row: tl.constexpr,
    stride_l2_plane: tl.constexpr, stride_l2_row: tl.constexpr,
    stride_l3_plane: tl.constexpr, stride_l3_row: tl.constexpr,
    stride_l4_plane: tl.constexpr, stride_l4_row: tl.constexpr,
    stride_l5_plane: tl.constexpr, stride_l5_row: tl.constexpr,
    stride_out_plane: tl.constexpr, stride_out_row: tl.constexpr,
    # Add tensor strides (used when HAS_ADD=True)
    stride_add_plane: tl.constexpr, stride_add_row: tl.constexpr,
    # Constants
    N: tl.constexpr,  # B*C*H1*W1
    LEVELS: tl.constexpr,
    HAS_ADD: tl.constexpr,  # Whether to fuse addition
    BLOCK_SIZE: tl.constexpr,
):
    """
    iHaar cascade kernel: 4 outputs per thread for better compute/load ratio.
    Grid is over B*C*H1*W1 (Level 1 coefficients).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode L1 coordinates
    x1 = offs % W1
    tmp = offs // W1
    y1 = tmp % H1
    bc = tmp // H1
    
    # Accumulator for cascade LL
    ll_curr = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # LEVEL 5
    if LEVELS >= 5:
        x5 = x1 >> 4
        y5 = y1 >> 4
        valid5 = mask & (y5 < H5) & (x5 < W5)
        
        off_bc5 = (bc * 4 * H5 * W5).to(tl.int64)
        base5 = l5_ptr + off_bc5
        idx5 = (y5 * stride_l5_row + x5).to(tl.int64)
        ps = stride_l5_plane
        
        l5_ll = tl.load(base5 + 0*ps + idx5, mask=valid5, other=0.0)
        l5_lh = tl.load(base5 + 1*ps + idx5, mask=valid5, other=0.0)
        l5_hl = tl.load(base5 + 2*ps + idx5, mask=valid5, other=0.0)
        l5_hh = tl.load(base5 + 3*ps + idx5, mask=valid5, other=0.0)
        
        qx = (x1 >> 3) & 1
        qy = (y1 >> 3) & 1
        
        lh_s = tl.where(qy == 0, l5_lh, -l5_lh)
        hh_s = tl.where(qy == 0, l5_hh, -l5_hh)
        term1 = l5_ll + lh_s
        term2 = l5_hl + hh_s
        term2_s = tl.where(qx == 0, term2, -term2)
        ll_curr = 0.5 * term1 + 0.5 * term2_s

    # LEVEL 4
    if LEVELS >= 4:
        x4 = x1 >> 3
        y4 = y1 >> 3
        valid4 = mask & (y4 < H4) & (x4 < W4)
        
        off_bc4 = (bc * 4 * H4 * W4).to(tl.int64)
        base4 = l4_ptr + off_bc4
        idx4 = (y4 * stride_l4_row + x4).to(tl.int64)
        ps = stride_l4_plane
        
        l4_ll = tl.load(base4 + 0*ps + idx4, mask=valid4, other=0.0)
        l4_lh = tl.load(base4 + 1*ps + idx4, mask=valid4, other=0.0)
        l4_hl = tl.load(base4 + 2*ps + idx4, mask=valid4, other=0.0)
        l4_hh = tl.load(base4 + 3*ps + idx4, mask=valid4, other=0.0)
        
        l4_ll = l4_ll + ll_curr
        qx = (x1 >> 2) & 1
        qy = (y1 >> 2) & 1
        
        lh_s = tl.where(qy == 0, l4_lh, -l4_lh)
        hh_s = tl.where(qy == 0, l4_hh, -l4_hh)
        term1 = l4_ll + lh_s
        term2 = l4_hl + hh_s
        term2_s = tl.where(qx == 0, term2, -term2)
        ll_curr = 0.5 * term1 + 0.5 * term2_s

    # LEVEL 3
    if LEVELS >= 3:
        x3 = x1 >> 2
        y3 = y1 >> 2
        valid3 = mask & (y3 < H3) & (x3 < W3)
        
        off_bc3 = (bc * 4 * H3 * W3).to(tl.int64)
        base3 = l3_ptr + off_bc3
        idx3 = (y3 * stride_l3_row + x3).to(tl.int64)
        ps = stride_l3_plane
        
        l3_ll = tl.load(base3 + 0*ps + idx3, mask=valid3, other=0.0)
        l3_lh = tl.load(base3 + 1*ps + idx3, mask=valid3, other=0.0)
        l3_hl = tl.load(base3 + 2*ps + idx3, mask=valid3, other=0.0)
        l3_hh = tl.load(base3 + 3*ps + idx3, mask=valid3, other=0.0)
        
        l3_ll = l3_ll + ll_curr
        qx = (x1 >> 1) & 1
        qy = (y1 >> 1) & 1
        
        lh_s = tl.where(qy == 0, l3_lh, -l3_lh)
        hh_s = tl.where(qy == 0, l3_hh, -l3_hh)
        term1 = l3_ll + lh_s
        term2 = l3_hl + hh_s
        term2_s = tl.where(qx == 0, term2, -term2)
        ll_curr = 0.5 * term1 + 0.5 * term2_s

    # LEVEL 2
    if LEVELS >= 2:
        x2 = x1 >> 1
        y2 = y1 >> 1
        valid2 = mask & (y2 < H2) & (x2 < W2)
        
        off_bc2 = (bc * 4 * H2 * W2).to(tl.int64)
        base2 = l2_ptr + off_bc2
        idx2 = (y2 * stride_l2_row + x2).to(tl.int64)
        ps = stride_l2_plane
        
        l2_ll = tl.load(base2 + 0*ps + idx2, mask=valid2, other=0.0)
        l2_lh = tl.load(base2 + 1*ps + idx2, mask=valid2, other=0.0)
        l2_hl = tl.load(base2 + 2*ps + idx2, mask=valid2, other=0.0)
        l2_hh = tl.load(base2 + 3*ps + idx2, mask=valid2, other=0.0)
        
        l2_ll = l2_ll + ll_curr
        qx = x1 & 1
        qy = y1 & 1
        
        lh_s = tl.where(qy == 0, l2_lh, -l2_lh)
        hh_s = tl.where(qy == 0, l2_hh, -l2_hh)
        term1 = l2_ll + lh_s
        term2 = l2_hl + hh_s
        term2_s = tl.where(qx == 0, term2, -term2)
        ll_curr = 0.5 * term1 + 0.5 * term2_s

    # LEVEL 1 - Full reconstruction (all 4 outputs)
    off_bc1 = (bc * 4 * H1 * W1).to(tl.int64)
    base1 = l1_ptr + off_bc1
    idx1 = (y1 * stride_l1_row + x1).to(tl.int64)
    ps = stride_l1_plane
    
    l1_ll = tl.load(base1 + 0*ps + idx1, mask=mask, other=0.0)
    l1_lh = tl.load(base1 + 1*ps + idx1, mask=mask, other=0.0)
    l1_hl = tl.load(base1 + 2*ps + idx1, mask=mask, other=0.0)
    l1_hh = tl.load(base1 + 3*ps + idx1, mask=mask, other=0.0)
    
    if LEVELS >= 2:
        l1_ll = l1_ll + ll_curr
    
    out00, out01, out10, out11 = _ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh)
    
    # OUTPUT STORES (4 per thread)
    y_out = y1 * 2
    x_out = x1 * 2
    out_off_bc = (bc * stride_out_plane).to(tl.int64)
    out_base = output_ptr + out_off_bc
    rs = stride_out_row
    
    idx00 = (y_out * rs + x_out).to(tl.int64)
    
    # Fuse addition if add_ptr provided
    if HAS_ADD:
        add_off_bc = (bc * stride_add_plane).to(tl.int64)
        add_base = add_ptr + add_off_bc
        add_rs = stride_add_row
        add_idx00 = (y_out * add_rs + x_out).to(tl.int64)
        
        add00 = tl.load(add_base + add_idx00, mask=mask, other=0.0)
        add01 = tl.load(add_base + add_idx00 + 1, mask=mask, other=0.0)
        add10 = tl.load(add_base + add_idx00 + add_rs, mask=mask, other=0.0)
        add11 = tl.load(add_base + add_idx00 + add_rs + 1, mask=mask, other=0.0)
        
        out00 = out00 + add00
        out01 = out01 + add01
        out10 = out10 + add10
        out11 = out11 + add11
    
    tl.store(out_base + idx00, out00, mask=mask)
    tl.store(out_base + idx00 + 1, out01, mask=mask)
    tl.store(out_base + idx00 + rs, out10, mask=mask)
    tl.store(out_base + idx00 + rs + 1, out11, mask=mask)


def run_ihaar_cascade(levels_list, output_size=None, add_tensor=None):
    """Dispatcher for iHaar cascade.
    
    Args:
        levels_list: List of level tensors [(B, C, 4, H1, W1), ...]
        output_size: Optional (H, W) tuple for output size
        add_tensor: Optional (B, C, H, W) tensor to fuse into output (eliminates separate add)
    """
    num_levels = len(levels_list)
    assert 1 <= num_levels <= 5
    
    l1 = levels_list[0]
    B, C = l1.shape[:2]
    H1, W1 = l1.shape[3], l1.shape[4]
    
    if output_size is None:
        H, W = H1 * 2, W1 * 2
    else:
        H, W = output_size
    
    output = torch.empty(B, C, H, W, device=l1.device, dtype=l1.dtype)
    
    ptrs = [l1] * 5
    strides = [0] * 10
    dims = [0] * 10
    
    for i, l in enumerate(levels_list):
        ptrs[i] = l
        s = l.stride()
        strides[2*i] = s[2]
        strides[2*i+1] = s[3]
        dims[2*i] = l.shape[3]
        dims[2*i+1] = l.shape[4]
    
    N = B * C * H1 * W1
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    out_stride_plane = output.stride(1)
    out_stride_row = output.stride(2)
    
    # Handle optional add tensor
    has_add = add_tensor is not None
    if has_add:
        add_ptr = add_tensor
        add_stride_plane = add_tensor.stride(1)
        add_stride_row = add_tensor.stride(2)
    else:
        add_ptr = output  # Dummy pointer (won't be used)
        add_stride_plane = 0
        add_stride_row = 0
    
    _ihaar_cascade_kernel[grid](
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4],
        output,
        add_ptr,
        H, W,
        dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], dims[8], dims[9],
        strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], strides[9],
        out_stride_plane, out_stride_row,
        add_stride_plane, add_stride_row,
        N, num_levels, has_add
    )
    
    return output


# Import forward Haar kernel for backward pass
try:
    from fast_wtconv.triton_haar.triton_haar import _compute_haar_coeffs_kernel
except ImportError:
    # Fallback for local testing
    from .triton_haar import _compute_haar_coeffs_kernel

def run_haar_cascade(x, num_levels):
    """
    Run forward Haar cascade (Backward of Inverse Haar).
    Computes coefficients for each level by iteratively calling single-level kernel.
    
    Args:
        x: Input tensor (B, C, H, W) of gradients
        num_levels: Number of levels to compute
        
    Returns:
        List of tensors [level1, level2, ..., level_n]
    """
    levels = []
    curr_x = x
    
    for i in range(num_levels):
        B, C, H, W = curr_x.shape
        H2, W2 = H // 2, W // 2
        
        # Output for this level: (B, C*4, H2, W2)
        # Layout: LL, LH, HL, HH interleved in channel dim
        output = torch.empty(B, C * 4, H2, W2, device=x.device, dtype=x.dtype)
        
        N = B * C * H2 * W2
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        
        _compute_haar_coeffs_kernel[grid](
            curr_x, output,
            B, C, H, W, H2, W2, N,
            curr_x.stride(0), curr_x.stride(1), curr_x.stride(2), curr_x.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        )
        
        # Reshape to (B, C, 4, H2, W2) for consistency with API
        level_out = output.view(B, C, 4, H2, W2)
        levels.append(level_out)
        
        # Next level input is the LL subband (channel 0)
        # We need (B, C, H2, W2) contiguous for best performance, but strided works too via kernel.
        # level_out[:, :, 0, :, :] is (B, C, H2, W2)
        if i < num_levels - 1:
            curr_x = level_out[:, :, 0, :, :]

    return levels


@torch._dynamo.allow_in_graph
class InverseHaarCascadeFn(torch.autograd.Function):
    """
    Autograd function for Inverse Haar Cascade.
    
    Forward: (Coeffs1, Coeffs2...) -> Image (Triton)
    Backward: Grad_Image -> (Grad_Coeffs1, Grad_Coeffs2...) (Triton - Forward Cascade)
    """
    
    @staticmethod
    def forward(ctx, output_size, *levels):
        # levels is a tuple of tensors
        output = run_ihaar_cascade(list(levels), output_size)
        ctx.sizes = [l.shape for l in levels]
        ctx.num_levels = len(levels)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grads = run_haar_cascade(grad_output, ctx.num_levels)
        
        # Ensure grads match input shapes (mainly for safety)
        # run_haar_cascade returns [L1, L2...]
        # We need to return (None, *grads) because first arg is output_size
        return (None, *grads)


@torch._dynamo.allow_in_graph
class InverseHaarCascadeFusedFn(torch.autograd.Function):
    """
    Autograd function for Inverse Haar Cascade with fused addition.
    
    Forward: (Coeffs1, Coeffs2..., add_tensor) -> Image + add_tensor (Triton, fused)
    Backward: Grad_Output -> (Grad_Coeffs1, ..., Grad_add_tensor)
    
    The gradient for add_tensor is simply grad_output (identity).
    """
    
    @staticmethod
    def forward(ctx, output_size, add_tensor, *levels):
        # levels is a tuple of tensors, add_tensor is the tensor to fuse
        output = run_ihaar_cascade(list(levels), output_size, add_tensor=add_tensor)
        ctx.sizes = [l.shape for l in levels]
        ctx.num_levels = len(levels)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grads = run_haar_cascade(grad_output, ctx.num_levels)
        
        # Gradient for add_tensor is just grad_output (d(x+y)/dy = 1)
        grad_add = grad_output
        
        # Return: (None for output_size, grad_add, *grads for levels)
        return (None, grad_add, *grads)


# Public API (cuda_haar compatible)

def ihaar2d(x, output_size=None):
    if output_size is None:
        B, C, _, H2, W2 = x.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFn.apply(output_size, x)

def ihaar2d_double(l1, l2, output_size=None):
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFn.apply(output_size, l1, l2)

def ihaar2d_triple(l1, l2, l3, output_size=None):
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFn.apply(output_size, l1, l2, l3)

def ihaar2d_quad(l1, l2, l3, l4, output_size=None):
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFn.apply(output_size, l1, l2, l3, l4)

def ihaar2d_quint(l1, l2, l3, l4, l5, output_size=None):
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFn.apply(output_size, l1, l2, l3, l4, l5)


# Fused API - add_tensor is fused into the output (eliminates separate add)

def ihaar2d_fused(x, add_tensor, output_size=None):
    """iHaar with fused addition: returns ihaar(x) + add_tensor."""
    if output_size is None:
        B, C, _, H2, W2 = x.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFusedFn.apply(output_size, add_tensor, x)

def ihaar2d_double_fused(l1, l2, add_tensor, output_size=None):
    """iHaar (2 levels) with fused addition."""
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFusedFn.apply(output_size, add_tensor, l1, l2)

def ihaar2d_triple_fused(l1, l2, l3, add_tensor, output_size=None):
    """iHaar (3 levels) with fused addition."""
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFusedFn.apply(output_size, add_tensor, l1, l2, l3)

def ihaar2d_quad_fused(l1, l2, l3, l4, add_tensor, output_size=None):
    """iHaar (4 levels) with fused addition."""
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFusedFn.apply(output_size, add_tensor, l1, l2, l3, l4)

def ihaar2d_quint_fused(l1, l2, l3, l4, l5, add_tensor, output_size=None):
    """iHaar (5 levels) with fused addition."""
    if output_size is None:
        B, C, _, H2, W2 = l1.shape
        output_size = (H2*2, W2*2)
    return InverseHaarCascadeFusedFn.apply(output_size, add_tensor, l1, l2, l3, l4, l5)


