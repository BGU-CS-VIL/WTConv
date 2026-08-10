// =============================================================================
// Depthwise convolution weight gradient.
//
//     dL/dW[c,kh,kw] = sum_{b,h,w} grad_out[b,c,h,w] * x[b,c, h+kh-R, w+kw-R]
//
// This is the same reduction as the wavelet weight gradient in
// fused_haar_conv.cu, minus the Haar step -- it exists because WTConv's base
// convolution is an ordinary depthwise conv, and cuDNN's grouped weight
// gradient for depthwise layers was the most expensive kernel left in a
// training step once the wavelet branch was fused.
//
// One block owns a channel and sweeps a long run of spatial tiles, holding K*K
// accumulators per thread in registers; the haloed input tile is staged in
// shared memory so each pixel is read from HBM once. Each warp finishes with a
// shuffle reduction and K*K atomic adds into the (tiny) fp32 output.
//
// The atomics make this non-deterministic at fp32 rounding level, as cuDNN's
// own weight gradient is.
// =============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "haar_common.cuh"

#define DW_TILE_W 32
#define DW_TILE_H 8
#define DW_THREADS (DW_TILE_W * DW_TILE_H)

template<typename T, int K>
__global__ __launch_bounds__(DW_THREADS) void depthwise_grad_weight_kernel(
    const T* __restrict__ input,         // (B, C, H, W), contiguous
    const T* __restrict__ grad_output,   // (B, C, H, W), contiguous
    float* __restrict__ grad_weight,     // (C, K, K) fp32, pre-zeroed
    int B, int C, int H, int W,
    int tiles_x, int tiles_area
) {
    constexpr int R = K / 2;
    constexpr int SH = DW_TILE_H + K - 1;
    constexpr int SW = DW_TILE_W + K - 1;
    constexpr int SPLANE = SH * SW;

    __shared__ float sh_x[SPLANE];

    float acc[K * K];
    #pragma unroll
    for (int i = 0; i < K * K; ++i) acc[i] = 0.f;

    const int c = blockIdx.y;
    const int tid = threadIdx.y * DW_TILE_W + threadIdx.x;
    const long tiles_total = (long)tiles_area * B;

    for (long tt = blockIdx.x; tt < tiles_total; tt += gridDim.x) {
        const int b = (int)(tt / tiles_area);
        const int tile = (int)(tt - (long)b * tiles_area);
        const int oh0 = (tile / tiles_x) * DW_TILE_H;
        const int ow0 = (tile % tiles_x) * DW_TILE_W;
        const size_t bc = (size_t)b * C + c;

        __syncthreads();   // previous iteration's reads must be done
        const T* in_bc = input + bc * H * W;
        for (int i = tid; i < SPLANE; i += DW_THREADS) {
            const int sy = i / SW;
            const int sx = i - sy * SW;
            const int ph = oh0 - R + sy;
            const int pw = ow0 - R + sx;
            sh_x[i] = (ph >= 0 && ph < H && pw >= 0 && pw < W)
                      ? to_float(__ldg(&in_bc[(size_t)ph * W + pw])) : 0.f;
        }
        __syncthreads();

        const int h = oh0 + threadIdx.y;
        const int w = ow0 + threadIdx.x;
        float g = 0.f;   // out-of-range positions contribute nothing
        if (h < H && w < W) {
            g = to_float(__ldg(&grad_output[bc * H * W + (size_t)h * W + w]));
        }

        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int srow = (threadIdx.y + kh) * SW + threadIdx.x;
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                acc[kh * K + kw] = fmaf(g, sh_x[srow + kw], acc[kh * K + kw]);
            }
        }
    }

    float* out = grad_weight + (size_t)c * K * K;
    #pragma unroll
    for (int i = 0; i < K * K; ++i) {
        float v = acc[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (threadIdx.x == 0) atomicAdd(out + i, v);
    }
}

// Only the sizes this build asked for are instantiated; see HAAR_MAX_K.
HAAR_DEFINE_LAUNCHER(depthwise_grad_weight_kernel)

// -----------------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------------
void depthwise_grad_weight(
    torch::Tensor input,          // (B, C, H, W)
    torch::Tensor grad_output,    // (B, C, H, W)
    torch::Tensor grad_weight     // (C, K, K) float32, zeroed
) {
    TORCH_CHECK(input.dim() == 4 && grad_output.dim() == 4,
                "input and grad_output must be (B, C, H, W)");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(input.sizes() == grad_output.sizes(),
                "input and grad_output must have the same shape (stride 1, 'same' padding)");
    TORCH_CHECK(input.scalar_type() == grad_output.scalar_type(),
                "input and grad_output must share dtype");
    TORCH_CHECK(grad_weight.is_cuda() && grad_weight.is_contiguous()
                && grad_weight.scalar_type() == torch::kFloat32,
                "grad_weight must be a contiguous float32 CUDA tensor");

    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    TORCH_CHECK(grad_weight.dim() == 3 && grad_weight.size(0) == C,
                "grad_weight must be (C, K, K)");
    const int K = (int)grad_weight.size(1);
    TORCH_CHECK(grad_weight.size(2) == K, "grad_weight must be square");
    // Whether this K was compiled is the dispatch's business, not this check's.
    TORCH_CHECK(K % 2 == 1 && K <= HAAR_K_LIMIT,
                "kernel_size must be odd and <= ", HAAR_K_LIMIT, ", got ", K);

    const int tiles_x = (W + DW_TILE_W - 1) / DW_TILE_W;
    const int tiles_y = (H + DW_TILE_H - 1) / DW_TILE_H;
    const int tiles_area = tiles_x * tiles_y;
    if (tiles_area == 0 || B == 0 || C == 0) return;

    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long tiles_total = (long)tiles_area * B;
    long bpc = (4L * sms + C - 1) / C;
    bpc = std::max(1L, std::min(bpc, tiles_total));

    dim3 block(DW_TILE_W, DW_TILE_H);
    dim3 grid((unsigned)bpc, (unsigned)C);
    auto stream = at::cuda::getCurrentCUDAStream();
    float* gwptr = grad_weight.data_ptr<float>();

    HAAR_DISPATCH_DTYPE(input, "depthwise_grad_weight", [&] {
        HAAR_DISPATCH_K(depthwise_grad_weight_kernel, scalar_t,
                        haar_cptr<scalar_t>(input), haar_cptr<scalar_t>(grad_output),
                        gwptr, B, C, H, W, tiles_x, tiles_area);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}
