"""
Throughput Benchmark: ConvNeXt vs WTConvNeXt

Measures inference throughput in images per second for:
- ConvNeXt-T/S/B (from timm)
- WTConvNeXt-T/S/B (original naive implementation)
- WTConvNeXt-T/S/B with CUDA kernels
- WTConvNeXt-T/S/B with Triton kernels

Configuration:
- Batch size: 64
- Input size: 224x224
- Warmup: 50 batches
- Measurement: 300 batches
"""

import sys
import os
import warnings
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton/cache")

# Suppress torch.compile warnings
warnings.filterwarnings("ignore", message=".*_maybe_guard_rel.*")
warnings.filterwarnings("ignore", message=".*recompile_limit.*")
warnings.filterwarnings("ignore", message=".*pow_by_natural*")
warnings.filterwarnings("ignore", message=".*evaluate_expr*")


from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import torch
import timm

# Suppress torch dynamo and symbolic shapes logging (after torch import)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
torch._dynamo.config.suppress_errors = True

# Add parent directory to path for custom wtconv implementations
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wtconvnext import wtconvnext_tiny, wtconvnext_small, wtconvnext_base

# =============================================================================
# Configuration flags
# =============================================================================
BENCHMARK_TRITON = True  # Set to True to include Triton benchmarks
BENCHMARK_CUDA = True  # Set to True to include CUDA benchmarks
BENCHMARK_REGULAR = True  # Set to True to include regular/naive WTConvNeXt benchmarks
USE_TORCH_COMPILE = True  # Set to True to wrap models with torch.compile() - incompatible with custom Triton kernels
CONVNEXT_KERNEL_SIZE = 7  # Kernel size for ConvNeXt depthwise convolutions (default: 7)
WTCONVNEXT_KERNEL_SIZE = 5  # Kernel size for WTConvNeXt depthwise convolutions (default: 5)
USE_CONV_MLP = True  # Use 1x1 conv in MLP


# Lazy-loaded WTConv classes
_WTConv2dCUDA = None
_WTConv2dTriton = None


def _get_wtconv_cuda():
    """Get CUDA WTConv2d class (lazy load to avoid compile messages during import)."""
    global _WTConv2dCUDA
    if _WTConv2dCUDA is None:
        # Suppress compilation output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from fast_wtconv.wtconv import WTConv2d
            _WTConv2dCUDA = WTConv2d
    return _WTConv2dCUDA


def _get_wtconv_triton():
    """Get Triton WTConv2d class."""
    global _WTConv2dTriton
    if _WTConv2dTriton is None:
        from fast_wtconv.wtconv_triton import WTConv2d
        _WTConv2dTriton = WTConv2d
    return _WTConv2dTriton


def create_wtconvnext_cuda(size='tiny'):
    """Create WTConvNeXt using CUDA Haar kernels."""
    WTConv2dCUDA = _get_wtconv_cuda()
    if size == 'tiny':
        return wtconvnext_tiny(pretrained=False, wtconv_class=WTConv2dCUDA, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    elif size == 'small':
        return wtconvnext_small(pretrained=False, wtconv_class=WTConv2dCUDA, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    else:
        return wtconvnext_base(pretrained=False, wtconv_class=WTConv2dCUDA, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)


def create_wtconvnext_triton(size='tiny'):
    """Create WTConvNeXt using Triton kernels."""
    WTConv2dTriton = _get_wtconv_triton()
    if size == 'tiny':
        return wtconvnext_tiny(pretrained=False, wtconv_class=WTConv2dTriton, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    elif size == 'small':
        return wtconvnext_small(pretrained=False, wtconv_class=WTConv2dTriton, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    else:
        return wtconvnext_base(pretrained=False, wtconv_class=WTConv2dTriton, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)


def benchmark_model(model, device, batch_size=64, warmup_batches=50, measure_batches=300):
    """
    Benchmark model throughput.
    
    Args:
        model: PyTorch model to benchmark
        device: CUDA device
        batch_size: Number of images per batch
        warmup_batches: Number of warmup batches (not timed)
        measure_batches: Number of batches to measure
        
    Returns:
        float: Throughput in images per second
    """
    model.eval()
    model = model.to(device)
    
    # Optionally compile the model
    if USE_TORCH_COMPILE:
        model = torch.compile(model)
    
    # Create input tensor
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Warmup (suppress output during first forward which may trigger compilation)
    with torch.no_grad():
        # First forward may trigger CUDA compilation - suppress output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _ = model(x)
        # Rest of warmup
        for _ in range(warmup_batches - 1):
            _ = model(x)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Measure
    with torch.no_grad():
        start_event.record()
        for _ in range(measure_batches):
            _ = model(x)
        end_event.record()
    
    # Wait for completion and get elapsed time
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_sec = elapsed_ms / 1000.0
    
    # Calculate throughput
    total_images = measure_batches * batch_size
    throughput = total_images / elapsed_sec
    
    return throughput


def main():
    device = torch.device('cuda')
    
    # Build model list based on configuration
    models = [
        # Tiny variants
        ('ConvNeXt-T', lambda: timm.create_model('convnext_tiny', pretrained=False, kernel_sizes=CONVNEXT_KERNEL_SIZE, conv_mlp=USE_CONV_MLP), True),
    ]
    
    if BENCHMARK_REGULAR:
        models.append(('WTConvNeXt-T', lambda: wtconvnext_tiny(pretrained=False, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE), False))
    
    if BENCHMARK_CUDA:
        models.append(('WTConvNeXt-T (CUDA)', lambda: create_wtconvnext_cuda('tiny'), False))
    
    if BENCHMARK_TRITON:
        models.append(('WTConvNeXt-T (Triton)', lambda: create_wtconvnext_triton('tiny'), False))
    
    models.extend([
        ('ConvNeXt-S', lambda: timm.create_model('convnext_small', pretrained=False, kernel_sizes=CONVNEXT_KERNEL_SIZE, conv_mlp=USE_CONV_MLP), True),
    ])
    
    if BENCHMARK_REGULAR:
        models.append(('WTConvNeXt-S', lambda: wtconvnext_small(pretrained=False, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE), False))
    
    if BENCHMARK_CUDA:
        models.append(('WTConvNeXt-S (CUDA)', lambda: create_wtconvnext_cuda('small'), False))
    
    if BENCHMARK_TRITON:
        models.append(('WTConvNeXt-S (Triton)', lambda: create_wtconvnext_triton('small'), False))
    
    # Base variants  
    models.extend([
        ('ConvNeXt-B', lambda: timm.create_model('convnext_base', pretrained=False, kernel_sizes=CONVNEXT_KERNEL_SIZE, conv_mlp=USE_CONV_MLP), True),
    ])
    
    if BENCHMARK_REGULAR:
        models.append(('WTConvNeXt-B', lambda: wtconvnext_base(pretrained=False, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE), False))
    
    if BENCHMARK_CUDA:
        models.append(('WTConvNeXt-B (CUDA)', lambda: create_wtconvnext_cuda('base'), False))
    
    if BENCHMARK_TRITON:
        models.append(('WTConvNeXt-B (Triton)', lambda: create_wtconvnext_triton('base'), False))
    
    print("\n" + "=" * 55)
    print("Throughput Benchmark (images per second)")
    print("=" * 55)
    print(f"\n{'Model':<25} {'Images/sec':>15}")
    print("-" * 42)
    
    results = []
    baseline_throughput = None
    current_size = None
    
    for name, model_factory, is_baseline in models:
        # Detect size change for grouping
        if 'ConvNeXt-T' in name and 'WT' not in name:
            current_size = 'T'
        elif 'ConvNeXt-S' in name and 'WT' not in name:
            current_size = 'S'
        elif 'ConvNeXt-B' in name and 'WT' not in name:
            current_size = 'B'

        model = model_factory()
        throughput = benchmark_model(model, device)
        results.append((name, throughput, is_baseline))
        
        if is_baseline:
            baseline_throughput = throughput
        
        # Calculate ratio if we have a baseline
        if baseline_throughput and not is_baseline:
            ratio = throughput / baseline_throughput * 100
            print(f"{name:<25} {throughput:>10.2f}  ({ratio:>5.1f}%)")
        else:
            print(f"{name:<25} {throughput:>10.2f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        
        # Add separator at end of each size group
        is_last_in_group = (
            (not BENCHMARK_TRITON and 'CUDA' in name) or
            (BENCHMARK_TRITON and 'Triton' in name)
        )
        if is_last_in_group:
            print("-" * 42)
    
    print("\n" + "=" * 55)
    print("Summary: Implementation Comparison by Model Size")
    print("=" * 55)
    
    # Group results by size
    for size in ['T', 'S', 'B']:
        convnext = next((r for r in results if r[0] == f'ConvNeXt-{size}'), None)
        if convnext:
            base_throughput = convnext[1]
            print(f"\n{size} variants (baseline: ConvNeXt-{size} = {base_throughput:.2f} img/sec)")
            
            for name, throughput, _ in results:
                if f'WTConvNeXt-{size}' in name and throughput > 0:
                    ratio = throughput / base_throughput * 100
                    print(f"  {name:<25}: {throughput:>8.2f} img/sec ({ratio:>5.1f}%)")


if __name__ == '__main__':
    main()
