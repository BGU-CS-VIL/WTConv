#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// Haar 2D Wavelet Transform - Inverse & Inverse Backward
// Inverse: (B, C, 4, H/2, W/2) -> (B, C, H, W)
// Supports fp32, fp16, and bf16 data types
// =============================================================================

// -----------------------------------------------------------------------------
// Type conversion helpers for generic programming
// -----------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ float to_float(T val);

template<>
__device__ __forceinline__ float to_float(float val) { return val; }

template<>
__device__ __forceinline__ float to_float(__half val) { return __half2float(val); }

template<>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ T from_float(float val);

template<>
__device__ __forceinline__ float from_float(float val) { return val; }

template<>
__device__ __forceinline__ __half from_float(float val) { return __float2half(val); }

template<>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// -----------------------------------------------------------------------------
// Inverse Haar Transform: (B, C, 4, H/2, W/2) -> (B, C, H, W)
// Reconstructs original image from wavelet coefficients
// -----------------------------------------------------------------------------
template<typename T>
__global__ void haar2d_inverse_kernel(
    const T* __restrict__ input,   // (B*C, 4, H2, W2)
    T* __restrict__ output,          // (B*C, H, W)
    int H, int W, int H2, int W2
) {
    // Grid decode
    // Block size 32x8
    const int tiles_x = (W2 + 31) / 32;
    const int tiles_y = (H2 + 7) / 8;
    const int tiles_area = tiles_x * tiles_y;

    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int x = threadIdx.x + tile_x * blockDim.x;
    const int y = threadIdx.y + tile_y * blockDim.y;
    
    if (y < H2 && x < W2) {
        int plane = H2 * W2;
        int in_offset = bc * 4 * plane;
        int idx = y * W2 + x;
        
        float ll = to_float(input[in_offset + 0 * plane + idx]);
        float lh = to_float(input[in_offset + 1 * plane + idx]);
        float hl = to_float(input[in_offset + 2 * plane + idx]);
        float hh = to_float(input[in_offset + 3 * plane + idx]);
        
        // Inverse Haar: reconstruct 2x2 block
        float a = 0.5f * (ll + lh + hl + hh);
        float b = 0.5f * (ll + lh - hl - hh);
        float c = 0.5f * (ll - lh + hl - hh);
        float d = 0.5f * (ll - lh - hl + hh);
        
        int x0 = 2 * x, x1 = 2 * x + 1;
        int y0 = 2 * y, y1 = 2 * y + 1;
        int out_offset = bc * H * W;
        
        output[out_offset + y0 * W + x0] = from_float<T>(a);
        if (x1 < W) output[out_offset + y0 * W + x1] = from_float<T>(b);
        if (y1 < H) output[out_offset + y1 * W + x0] = from_float<T>(c);
        if (x1 < W && y1 < H) output[out_offset + y1 * W + x1] = from_float<T>(d);
    }
}

// -----------------------------------------------------------------------------
// Backward for Inverse Haar: (B, C, H, W) -> (B, C, 4, H/2, W/2)
// This is essentially the forward Haar transform
// -----------------------------------------------------------------------------
template<typename T>
__global__ void haar2d_inverse_backward_kernel(
    const T* __restrict__ grad_output,  // (B*C, H, W)
    T* __restrict__ grad_input,          // (B*C, 4, H2, W2)
    int H, int W, int H2, int W2
) {
    // Grid decode
    // Block size 32x8
    const int tiles_x = (W2 + 31) / 32;
    const int tiles_y = (H2 + 7) / 8;
    const int tiles_area = tiles_x * tiles_y;

    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int x = threadIdx.x + tile_x * blockDim.x;
    const int y = threadIdx.y + tile_y * blockDim.y;
    
    if (y < H2 && x < W2) {
        int x0 = 2 * x, x1 = min(2 * x + 1, W - 1);
        int y0 = 2 * y, y1 = min(2 * y + 1, H - 1);
        
        int grad_offset = bc * H * W;
        float a = to_float(grad_output[grad_offset + y0 * W + x0]);
        float b = to_float(grad_output[grad_offset + y0 * W + x1]);
        float c = to_float(grad_output[grad_offset + y1 * W + x0]);
        float d = to_float(grad_output[grad_offset + y1 * W + x1]);
        
        float sum_ac = a + c, sum_bd = b + d;
        float diff_ac = a - c, diff_bd = b - d;
        
        int out_idx = y * W2 + x;
        int plane = H2 * W2;
        int out_offset = bc * 4 * plane;
        
        grad_input[out_offset + 0 * plane + out_idx] = from_float<T>(0.5f * (sum_ac + sum_bd));   // LL
        grad_input[out_offset + 1 * plane + out_idx] = from_float<T>(0.5f * (diff_ac + diff_bd)); // LH
        grad_input[out_offset + 2 * plane + out_idx] = from_float<T>(0.5f * (sum_ac - sum_bd));   // HL
        grad_input[out_offset + 3 * plane + out_idx] = from_float<T>(0.5f * (diff_ac - diff_bd)); // HH
    }
}

// -----------------------------------------------------------------------------
// Host Wrappers with dtype dispatch
// -----------------------------------------------------------------------------
void haar2d_inverse(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (B, C, 4, H2, W2)");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    
    int B = input.size(0);
    int C = input.size(1);
    int H2 = input.size(3);
    int W2 = input.size(4);
    int H = output.size(2);
    int W = output.size(3);
    
    dim3 block(32, 8);
    int tiles_x = (W2 + 31) / 32;
    int tiles_y = (H2 + 7) / 8;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    if (input.dtype() == torch::kFloat32) {
        haar2d_inverse_kernel<float><<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            H, W, H2, W2
        );
    } else if (input.dtype() == torch::kFloat16) {
        haar2d_inverse_kernel<__half><<<grid, block>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            H, W, H2, W2
        );
    } else if (input.dtype() == torch::kBFloat16) {
        haar2d_inverse_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            H, W, H2, W2
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Supported: float32, float16, bfloat16");
    }
}

void haar2d_inverse_backward(torch::Tensor grad_output, torch::Tensor grad_input) {
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be 4D (B, C, H, W)");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on CUDA");
    
    int B = grad_output.size(0);
    int C = grad_output.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);
    int H2 = grad_input.size(3);
    int W2 = grad_input.size(4);
    
    dim3 block(32, 8);
    int tiles_x = (W2 + 31) / 32;
    int tiles_y = (H2 + 7) / 8;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    if (grad_output.dtype() == torch::kFloat32) {
        haar2d_inverse_backward_kernel<float><<<grid, block>>>(
            grad_output.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            H, W, H2, W2
        );
    } else if (grad_output.dtype() == torch::kFloat16) {
        haar2d_inverse_backward_kernel<__half><<<grid, block>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            H, W, H2, W2
        );
    } else if (grad_output.dtype() == torch::kBFloat16) {
        haar2d_inverse_backward_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            H, W, H2, W2
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Supported: float32, float16, bfloat16");
    }
}
