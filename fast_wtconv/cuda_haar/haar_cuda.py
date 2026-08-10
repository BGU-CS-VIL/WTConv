"""
Fused CUDA Haar kernels for WTConv.

Everything here is built around fusing the Haar transform into the depthwise
convolution weights, so the wavelet coefficients are never written to memory:

    fused_haar_conv_scale(x, weight, scale, K)   Haar -> conv -> scale, one kernel
    ihaar2d_*_fused(levels..., add)              1-5 level inverse cascade + add
    scaled_depthwise_conv(x, w, s, pad, bias)    base-conv path (scale folded, cuDNN)

Layout conventions:
    coefficients   (B, C, 4, H/2, W/2) contiguous, subbands [LL, LH, HL, HH]
    conv weights   (C*4, 1, K, K), channel c*4+s holds subband s of channel c
    scales         (1, C*4, 1, 1)
    fused weights  (C, 4, K, K) float32 = scale * weight, built by
                   compute_scaled_weight()
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


# =============================================================================
# Extension loading
# =============================================================================

def _setup_cuda_arch():
    """Auto-detect the compute capability so nvcc does not warn."""
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                arch = result.stdout.strip().split('\n')[0]
                os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        except Exception:
            pass


_setup_cuda_arch()

_module = None

# Hard ceiling of the kernels themselves (HAAR_K_LIMIT in haar_common.cuh): past
# K = 29 their static shared memory exceeds the 48 KiB per-block limit.
HARD_MAX_KERNEL_SIZE = 29

# Default build ceiling. Every odd K up to the ceiling is instantiated for all
# three dtypes, and the weight-gradient kernels hold K*K accumulators in a fully
# unrolled register array, so compile time grows superlinearly in K: the whole
# range up to 29 takes minutes. Nearly all use is K <= 7, so that is the default
# and anything larger is an opt-in rebuild.
DEFAULT_MAX_KERNEL_SIZE = 7


def _read_max_kernel_size() -> int:
    """Build ceiling from $HAAR_MAX_K, defaulting to DEFAULT_MAX_KERNEL_SIZE."""
    raw = os.environ.get('HAAR_MAX_K')
    if raw is None or raw == '':
        return DEFAULT_MAX_KERNEL_SIZE
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(f"HAAR_MAX_K must be an integer, got {raw!r}") from None
    if k < 1 or k > HARD_MAX_KERNEL_SIZE or k % 2 == 0:
        raise ValueError(
            f"HAAR_MAX_K must be odd and in [1, {HARD_MAX_KERNEL_SIZE}], got {k}"
        )
    return k


# Largest kernel size this build instantiates. Anything larger is rejected, not
# routed elsewhere.
MAX_KERNEL_SIZE = _read_max_kernel_size()


def check_kernel_size(kernel_size: int) -> None:
    """
    Reject kernel sizes this build cannot run, naming the knob that fixes it.

    Called before the extension is loaded, so asking for a K outside the build
    fails at once instead of after a compile that cannot serve the request.
    """
    K = kernel_size
    if K < 1 or K % 2 != 1:
        raise ValueError(f"kernel_size must be odd and positive, got {K}")
    if K > HARD_MAX_KERNEL_SIZE:
        raise ValueError(
            f"kernel_size must be <= {HARD_MAX_KERNEL_SIZE}, got {K}"
        )
    if K > MAX_KERNEL_SIZE:
        raise ValueError(
            f"kernel_size {K} was not compiled into this build "
            f"(HAAR_MAX_K={MAX_KERNEL_SIZE}). Rerun with HAAR_MAX_K={K} to build "
            f"it -- a one-time compile, cached separately per ceiling."
        )


def _get_module():
    global _module
    if _module is None:
        src_dir = Path(__file__).parent
        # Each ceiling gets its own cached build, so switching between them does
        # not invalidate the others.
        name = f'fused_wtconv_haar_k{MAX_KERNEL_SIZE}'
        print(f"Compiling fused Haar CUDA kernels", flush=True)
        _module = load(
            name=name,
            sources=[
                str(src_dir / 'haar.cpp'),
                str(src_dir / 'fused_haar_conv.cu'),
                str(src_dir / 'ihaar_cascade.cu'),
                str(src_dir / 'depthwise_grad.cu'),
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               f'-DHAAR_MAX_K={MAX_KERNEL_SIZE}'],
            verbose=bool(os.environ.get('HAAR_BUILD_VERBOSE')),
        )
        print("Done.", flush=True)
    return _module


# =============================================================================
# Fused Haar -> conv -> scale
# =============================================================================

def compute_scaled_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Fold the per-channel scale into the depthwise weights and regroup them by
    subband, giving the (C, 4, K, K) float32 tensor the fused kernel consumes.

    Args:
        weight: (C*4, 1, K, K) depthwise conv weights
        scale: (1, C*4, 1, 1) or (C*4,) per-channel scales
        kernel_size: K

    Returns:
        (C, 4, K, K) float32, contiguous
    """
    C4 = weight.shape[0]
    K = kernel_size
    scaled = weight.reshape(C4, K, K) * scale.reshape(C4, 1, 1)
    return scaled.reshape(C4 // 4, 4, K, K).to(torch.float32).contiguous()


def _haar_coeffs(x: torch.Tensor) -> torch.Tensor:
    """Single-level Haar coefficients: (B, C, H, W) -> (B, C, 4, ceil(H/2), ceil(W/2))."""
    B, C, H, W = x.shape
    out = torch.empty(B, C, 4, (H + 1) // 2, (W + 1) // 2,
                      device=x.device, dtype=x.dtype)
    _get_module().haar_coeffs(x, out)
    return out


def _grad_weight_scale(
    level_input: torch.Tensor,  # (B, C, H, W) the (padded) input of this level
    grad_output: torch.Tensor,  # (B, C, 4, H2, W2)
    weight: torch.Tensor,       # (C*4, 1, K, K)
    scale: torch.Tensor,        # (1, C*4, 1, 1)
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of the *unfused* weight and scale.

    The forward convolves with w~ = scale * weight, so the chain rule gives
        dL/dweight = scale * dL/dw~        dL/dscale = sum(weight * dL/dw~).

    dL/dw~ comes from a kernel that reduces grad_output against the Haar partial
    sums computed on the fly, so the coefficients are never materialised.
    """
    K = kernel_size
    C = level_input.shape[1]
    C4 = C * 4

    grad_fused = torch.zeros(C, 4, K, K, device=level_input.device,
                             dtype=torch.float32)
    _get_module().fused_haar_grad_weight(
        level_input, grad_output.contiguous(), grad_fused
    )
    grad_fused = grad_fused.reshape(C4, 1, K, K).to(weight.dtype)

    grad_weight = grad_fused * scale.reshape(C4, 1, 1, 1)
    grad_scale = (grad_fused * weight).sum(dim=(1, 2, 3)).reshape_as(scale)
    return grad_weight, grad_scale


class FusedHaarConvScaleFunction(Function):
    """
    Autograd wrapper around the fused Haar -> conv -> scale kernel.

    forward:  x (B, C, H, W) -> coeffs (B, C, 4, H/2, W/2) [+ raw LL]
    backward: one fused kernel for grad_input, another for grad_weight /
              grad_scale; neither materialises the coefficients.
    """

    @staticmethod
    def forward(ctx, x, weight, scale, kernel_size, return_ll):
        assert x.is_cuda, "input must be on CUDA"
        assert x.dim() == 4, "input must be (B, C, H, W)"
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, \
            f"fused Haar conv needs even spatial dims, got {H}x{W} (pad first)"
        K = kernel_size
        check_kernel_size(K)

        x = x.contiguous()
        H2, W2 = H // 2, W // 2

        output = torch.empty(B, C, 4, H2, W2, device=x.device, dtype=x.dtype)
        ll_output = torch.empty(B, C, H2, W2, device=x.device, dtype=x.dtype) \
            if return_ll else None

        fused_weight = compute_scaled_weight(weight, scale, K)
        _get_module().fused_haar_conv_forward(x, fused_weight, output, ll_output)

        ctx.save_for_backward(x, weight, scale, fused_weight)
        ctx.kernel_size = K
        ctx.return_ll = return_ll

        if return_ll:
            return output, ll_output
        return output

    @staticmethod
    def backward(ctx, grad_output, grad_ll=None):
        x, weight, scale, fused_weight = ctx.saved_tensors
        K = ctx.kernel_size
        B, C, H, W = x.shape

        grad_input = grad_weight = grad_scale = None
        need_x, need_w, need_s = ctx.needs_input_grad[:3]

        grad_output = grad_output.contiguous()

        if need_x:
            grad_input = torch.empty_like(x)
            if grad_ll is not None:
                grad_ll = grad_ll.contiguous()
            _get_module().fused_haar_conv_backward(
                grad_output, fused_weight, grad_input, grad_ll
            )

        if need_w or need_s:
            grad_weight, grad_scale = _grad_weight_scale(
                x, grad_output, weight, scale, K
            )
            if not need_w:
                grad_weight = None
            if not need_s:
                grad_scale = None

        return grad_input, grad_weight, grad_scale, None, None


def fused_haar_conv_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int = 3,
    return_ll: bool = False,
):
    """
    Fused Haar transform -> depthwise conv -> scale, in a single kernel.

    Args:
        x: (B, C, H, W) with even H, W
        weight: (C*4, 1, K, K) depthwise conv weights
        scale: (1, C*4, 1, 1) per-channel scales
        kernel_size: K (odd, <= MAX_KERNEL_SIZE)
        return_ll: also return the raw LL subband (B, C, H/2, W/2), i.e. the
                   input of the next decomposition level, computed for free.

    Returns:
        coeffs: (B, C, 4, H/2, W/2)
        ll_raw: (B, C, H/2, W/2) when return_ll=True
    """
    return FusedHaarConvScaleFunction.apply(x, weight, scale, kernel_size, return_ll)


# =============================================================================
# Whole wavelet branch: every level plus the inverse cascade
# =============================================================================

def _pad_even(x: torch.Tensor) -> torch.Tensor:
    """Zero pad odd spatial dims to even, as the reference does at each level."""
    h, w = x.shape[2], x.shape[3]
    if (h & 1) or (w & 1):
        return F.pad(x, (0, w & 1, 0, h & 1))
    return x


class WaveletBranchFunction(Function):
    """
    The complete WTConv wavelet branch as a single autograd node.

    Every level runs the coefficient-producing fused kernel -- Haar, conv and
    scale in one pass -- emitting its filtered coefficients for the cascade plus,
    except at the deepest level, the raw LL the next level decomposes. The
    inverse cascade then reconstructs all levels in one kernel and folds the
    base-conv addition into its final store, so no intermediate low-pass or
    reconstruction tensor is ever materialised.

    Owning the whole branch is what makes the last part possible: splitting the
    levels across separate autograd nodes would force the raw-LL gradient into a
    separate full-resolution add; here it folds into the same grad-input kernel.
    """

    @staticmethod
    def forward(ctx, x, base_out, kernel_size, num_levels, *params):
        weights = params[:num_levels]
        scales = params[num_levels:]
        assert len(scales) == num_levels
        assert x.is_cuda and x.dim() == 4, "input must be a 4D CUDA tensor"
        K = kernel_size
        check_kernel_size(K)

        mod = _get_module()
        B, C, H, W = x.shape
        fused_ws = [compute_scaled_weight(w, s, K) for w, s in zip(weights, scales)]

        current = x.contiguous()
        level_inputs = []
        level_coeffs = []

        for i in range(num_levels):
            padded = _pad_even(current)
            level_inputs.append(padded)
            h2, w2 = padded.shape[2] // 2, padded.shape[3] // 2
            coeffs = torch.empty(B, C, 4, h2, w2, device=x.device, dtype=x.dtype)
            ll_out = (torch.empty(B, C, h2, w2, device=x.device, dtype=x.dtype)
                      if i < num_levels - 1 else None)
            mod.fused_haar_conv_forward(padded, fused_ws[i], coeffs, ll_out)
            level_coeffs.append(coeffs)
            current = ll_out

        output = torch.empty(B, C, H, W, device=x.device, dtype=x.dtype)
        base_out_cont = base_out.contiguous() if base_out is not None else None
        mod.ihaar_cascade(level_coeffs, output, base_out_cont)

        ctx.save_for_backward(*level_inputs, *fused_ws, *weights, *scales)
        ctx.num_levels = num_levels
        ctx.kernel_size = K
        ctx.input_shape = (H, W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        L = ctx.num_levels
        K = ctx.kernel_size
        H, W = ctx.input_shape
        saved = ctx.saved_tensors
        level_inputs = saved[:L]
        fused_ws = saved[L:2 * L]
        weights = saved[2 * L:3 * L]
        scales = saved[3 * L:4 * L]

        mod = _get_module()
        grad_output = grad_output.contiguous()
        B, C = grad_output.shape[:2]

        needs = ctx.needs_input_grad
        need_x, need_base = needs[0], needs[1]
        need_w = [needs[4 + i] for i in range(L)]
        need_s = [needs[4 + L + i] for i in range(L)]

        # Adjoint of (crop . inverse Haar) at level 1: the forward Haar of the
        # zero-padded output gradient.
        x0 = level_inputs[0]
        H1, W1 = x0.shape[2] // 2, x0.shape[3] // 2
        g1 = torch.empty(B, C, 4, H1, W1, device=grad_output.device, dtype=grad_output.dtype)
        mod.haar_coeffs(grad_output, g1)

        # Deeper levels: the cascade's adjoint is the forward Haar cascade of the
        # gradient that reached its output, i.e. level 1's LL gradient.
        grad_levels = [g1]
        if L > 1:
            grad_levels += run_haar_cascade(g1[:, :, 0].contiguous(), L - 1)

        # Walk levels L..2 backwards; each level's grad_input becomes the next
        # (shallower) level's grad_ll. Only the input gradient needs this chain;
        # the weight gradients read grad_levels directly.
        grad_ll = None
        for i in range(L - 1, 0, -1) if need_x else ():
            grad_in = torch.empty_like(level_inputs[i])
            mod.fused_haar_conv_backward(grad_levels[i], fused_ws[i], grad_in, grad_ll)
            # Undo this level's even padding before handing it up
            prev_ll_h, prev_ll_w = level_inputs[i - 1].shape[2] // 2, level_inputs[i - 1].shape[3] // 2
            if grad_in.shape[2] != prev_ll_h or grad_in.shape[3] != prev_ll_w:
                grad_in = grad_in[:, :, :prev_ll_h, :prev_ll_w].contiguous()
            grad_ll = grad_in

        grad_x = None
        if need_x:
            # grad_ll (the raw-LL path feeding level 2) folds into the same
            # kernel as the level-1 coefficient gradients: no extra pass.
            grad_x_pad = torch.empty_like(x0)
            mod.fused_haar_conv_backward(g1, fused_ws[0], grad_x_pad, grad_ll)
            grad_x = grad_x_pad
            if grad_x.shape[2] != H or grad_x.shape[3] != W:
                grad_x = grad_x[:, :, :H, :W].contiguous()

        grad_weights, grad_scales = [None] * L, [None] * L
        for i in range(L):
            if need_w[i] or need_s[i]:
                gw, gs = _grad_weight_scale(
                    level_inputs[i], grad_levels[i], weights[i], scales[i], K)
                grad_weights[i] = gw if need_w[i] else None
                grad_scales[i] = gs if need_s[i] else None

        grad_base = grad_output if need_base else None
        return (grad_x, grad_base, None, None, *grad_weights, *grad_scales)


def wavelet_branch(
    x: torch.Tensor,
    base_out: torch.Tensor,
    weights: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    kernel_size: int,
) -> torch.Tensor:
    """
    Run WTConv's entire wavelet branch and add the base-conv output.

    Args:
        x: (B, C, H, W)
        base_out: (B, C, H, W) scaled base convolution, folded into the final store
        weights: per level, (C*4, 1, K, K)
        scales: per level, (1, C*4, 1, 1)
        kernel_size: K (odd, <= MAX_KERNEL_SIZE)

    Returns:
        (B, C, H, W)
    """
    num_levels = len(weights)
    assert len(scales) == num_levels, "one scale per level"
    assert 1 <= num_levels <= 5, "wt_levels must be 1-5"
    return WaveletBranchFunction.apply(
        x, base_out, kernel_size, num_levels, *weights, *scales
    )


# =============================================================================
# Inverse Haar cascade (1-5 levels, optional fused add)
# =============================================================================

def run_ihaar_cascade(
    levels: Sequence[torch.Tensor],
    output_size: Optional[Tuple[int, int]] = None,
    add_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused inverse Haar cascade.

    Args:
        levels: [(B, C, 4, H1, W1), (B, C, 4, H2, W2), ...] finest level first
        output_size: (H, W) of the reconstruction; defaults to (2*H1, 2*W1)
        add_tensor: optional (B, C, H, W) folded into the output store

    Returns:
        (B, C, H, W)
    """
    levels = [l.contiguous() for l in levels]
    assert 1 <= len(levels) <= 5, "cascade supports 1-5 levels"

    B, C = levels[0].shape[:2]
    H1, W1 = levels[0].shape[3], levels[0].shape[4]
    H, W = output_size if output_size is not None else (H1 * 2, W1 * 2)

    output = torch.empty(B, C, H, W, device=levels[0].device, dtype=levels[0].dtype)
    if add_tensor is not None:
        add_tensor = add_tensor.contiguous()
    _get_module().ihaar_cascade(levels, output, add_tensor)
    return output


def run_haar_cascade(x: torch.Tensor, num_levels: int) -> List[torch.Tensor]:
    """
    Forward Haar cascade -- the gradient of the inverse cascade.

    Each level transforms the previous level's LL subband, matching the
    ceil-halving shape chain of the forward pass.
    """
    levels = []
    curr = x
    for i in range(num_levels):
        coeffs = _haar_coeffs(curr)
        levels.append(coeffs)
        if i < num_levels - 1:
            curr = coeffs[:, :, 0, :, :]
    return levels


class IHaarCascadeFn(Function):
    """Inverse cascade; `add` may be None. Backward is the forward cascade."""

    @staticmethod
    def forward(ctx, output_size, add, *levels):
        ctx.num_levels = len(levels)
        ctx.has_add = add is not None
        ctx.level_shapes = [tuple(l.shape) for l in levels]
        return run_ihaar_cascade(list(levels), output_size, add)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grads = run_haar_cascade(grad_output, ctx.num_levels)
        for g, shape in zip(grads, ctx.level_shapes):
            assert tuple(g.shape) == shape, \
                f"inverse-cascade gradient shape {tuple(g.shape)} != level shape {shape}"
        grad_add = grad_output if ctx.has_add else None
        return (None, grad_add, *grads)


def _ihaar(levels, add, output_size):
    if output_size is None:
        H2, W2 = levels[0].shape[3], levels[0].shape[4]
        output_size = (H2 * 2, W2 * 2)
    return IHaarCascadeFn.apply(output_size, add, *levels)


# Plain inverse cascade -------------------------------------------------------

def ihaar2d(x, output_size=None):
    return _ihaar([x], None, output_size)


def ihaar2d_double(l1, l2, output_size=None):
    return _ihaar([l1, l2], None, output_size)


def ihaar2d_triple(l1, l2, l3, output_size=None):
    return _ihaar([l1, l2, l3], None, output_size)


def ihaar2d_quad(l1, l2, l3, l4, output_size=None):
    return _ihaar([l1, l2, l3, l4], None, output_size)


def ihaar2d_quint(l1, l2, l3, l4, l5, output_size=None):
    return _ihaar([l1, l2, l3, l4, l5], None, output_size)


# Inverse cascade with the final add fused in ---------------------------------

def ihaar2d_fused(x, add_tensor, output_size=None):
    """ihaar(x) + add_tensor, in one kernel."""
    return _ihaar([x], add_tensor, output_size)


def ihaar2d_double_fused(l1, l2, add_tensor, output_size=None):
    return _ihaar([l1, l2], add_tensor, output_size)


def ihaar2d_triple_fused(l1, l2, l3, add_tensor, output_size=None):
    return _ihaar([l1, l2, l3], add_tensor, output_size)


def ihaar2d_quad_fused(l1, l2, l3, l4, add_tensor, output_size=None):
    return _ihaar([l1, l2, l3, l4], add_tensor, output_size)


def ihaar2d_quint_fused(l1, l2, l3, l4, l5, add_tensor, output_size=None):
    return _ihaar([l1, l2, l3, l4, l5], add_tensor, output_size)


# =============================================================================
# Plain forward Haar (utility / testing)
# =============================================================================

class HaarTransform(Function):
    """Single-level Haar transform. The transform is orthogonal, so its
    gradient is the inverse transform."""

    @staticmethod
    def forward(ctx, x):
        ctx.shape_hw = (x.shape[2], x.shape[3])
        return _haar_coeffs(x.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        return run_ihaar_cascade([grad_output.contiguous()], ctx.shape_hw)


def haar2d(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, 4, ceil(H/2), ceil(W/2)), subbands [LL, LH, HL, HH]."""
    return HaarTransform.apply(x)


# =============================================================================
# Scaled depthwise conv (base-conv path): scale folded into weight and bias
# =============================================================================

def _depthwise_grad_weight(input, grad_output, weight, padding, groups):
    """
    Weight gradient of a depthwise conv, via the dedicated kernel. The layer must
    be the shape WTConv's base conv always is: depthwise, stride 1, 'same'
    padding, odd K <= MAX_KERNEL_SIZE.
    """
    C, K = weight.shape[0], weight.shape[2]
    assert groups == C and weight.shape[1] == 1, \
        f"conv must be depthwise, got groups={groups} for {tuple(weight.shape)}"
    assert weight.shape[3] == K, "kernel must be square"
    check_kernel_size(K)
    assert padding == K // 2, f"conv must use 'same' padding, got {padding} for K={K}"
    assert input.shape[2:] == grad_output.shape[2:], "conv must be stride 1"

    grad_w = torch.zeros(C, K, K, device=input.device, dtype=torch.float32)
    _get_module().depthwise_grad_weight(
        input.contiguous(), grad_output.contiguous(), grad_w
    )
    return grad_w.reshape(C, 1, K, K).to(weight.dtype)


class ScaledDepthwiseConvFunction(Function):
    """
    y = scale * conv2d(x, weight, bias), computed as conv2d(x, scale*weight,
    scale*bias) so cuDNN handles the forward and the input gradient; the weight
    gradient goes through the dedicated depthwise kernel.
    """

    @staticmethod
    def forward(ctx, input, weight, scale, bias, padding, groups):
        scale_flat = scale.reshape(-1)
        fused_weight = scale_flat.view(-1, 1, 1, 1) * weight
        fused_bias = None if bias is None else scale_flat * bias
        output = F.conv2d(input, fused_weight, bias=fused_bias,
                          padding=padding, groups=groups)

        saved_bias = input.new_empty(0) if bias is None else bias
        ctx.save_for_backward(input, weight, scale, fused_weight, saved_bias)
        ctx.padding = padding
        ctx.groups = groups
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, scale, fused_weight, saved_bias = ctx.saved_tensors
        padding, groups = ctx.padding, ctx.groups

        grad_input = torch.nn.grad.conv2d_input(
            input.shape, fused_weight, grad_output, padding=padding, groups=groups
        )
        grad_fused_weight = _depthwise_grad_weight(
            input, grad_output, weight, padding, groups
        )

        # Unfuse. The forward folds the scale into the weight, W~ = s * W, so the
        # chain rule carries that factor back: dL/dW = s * dL/dW~. Dropping it
        # leaves grad_weight wrong by a per-channel factor of s (it only happens
        # to be right while s == 1, i.e. at initialisation).
        grad_weight = grad_fused_weight * scale.view(-1, 1, 1, 1)
        grad_scale = (grad_fused_weight * weight).sum(dim=(1, 2, 3))

        if ctx.has_bias:
            grad_fused_bias = grad_output.sum(dim=(0, 2, 3))
            grad_bias = scale.reshape(-1) * grad_fused_bias
            grad_scale = grad_scale + saved_bias * grad_fused_bias
        else:
            grad_bias = None

        grad_scale = grad_scale.reshape_as(scale)
        return grad_input, grad_weight, grad_scale, grad_bias, None, None


def scaled_depthwise_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    padding: int = 1,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Scaled depthwise convolution: scale * depthwise_conv(input, weight, bias).

    Args:
        input: (B, C, H, W)
        weight: (C, 1, K, K)
        scale: (1, C, 1, 1)
        padding: usually kernel_size // 2
        bias: optional (C,), scaled along with the conv output
    """
    groups = input.size(1)
    return ScaledDepthwiseConvFunction.apply(input, weight, scale, bias, padding, groups)
