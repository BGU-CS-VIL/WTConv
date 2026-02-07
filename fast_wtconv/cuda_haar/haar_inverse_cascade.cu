#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// Optimized Fused Inverse Haar 2D - V2 (All Levels: 2, 3, 4, 5)
//
// Optimizations:
// 1. __ldg() for texture cache path (non-coherent loads)
// 2. 2x2 output per thread to amortize coefficient loads
// 3. Fast path template for even dimensions (no boundary checks)
// =============================================================================

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

__device__ __forceinline__ void ihaar_step(
    float ll, float lh, float hl, float hh,
    float& a, float& b, float& c, float& d
) {
    a = 0.5f * (ll + lh + hl + hh);
    b = 0.5f * (ll + lh - hl - hh);
    c = 0.5f * (ll - lh + hl - hh);
    d = 0.5f * (ll - lh - hl + hh);
}

template<typename T>
__device__ __forceinline__ void load_subbands_ldg(
    const T* __restrict__ data, int offset, int plane, int idx,
    float& ll, float& lh, float& hl, float& hh
) {
    ll = to_float(__ldg(&data[offset + 0 * plane + idx]));
    lh = to_float(__ldg(&data[offset + 1 * plane + idx]));
    hl = to_float(__ldg(&data[offset + 2 * plane + idx]));
    hh = to_float(__ldg(&data[offset + 3 * plane + idx]));
}

// Macro for output write with boundary check
#define WRITE_OUTPUT(y, x, val) \
    if (EVEN_DIMS || ((y) < H && (x) < W)) { \
        output[out_offset + (y) * W + (x)] = from_float<T>(val); \
    }

// =============================================================================
// 2-Level Inverse Cascade
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void ihaar2d_double_cascade_kernel(
    const T* __restrict__ level1, const T* __restrict__ level2,
    T* __restrict__ output, int H, int W, int H2, int W2, int H4, int W4
) {
    // Grid decode
    // Tiling based on output size H, W (16x16 output per thread block of 16x16? Wait. )
    // Forward double uses 16x16 threads for 4x4 input each -> 64x64 input? No.
    // Forward double: each thread 4x4 input.
    // Inverse double: 
    // const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    // const int x_base = tx * 2; -> Each thread produces 2x2.
    // Block (16,16). Threads 256. 
    // Output per block: 32x32 pixels.
    // Tiles needed: (W+31)/32?
    // Let's check original host code:
    // dim3 grid(((W + 1) / 2 + 15) / 16, ((H + 1) / 2 + 15) / 16, B * C);
    // Grid coords are in terms of "blocks of threads".
    // Each thread block covers 16 threads in X => 16*2 = 32 pixels in X.
    // So tiles_x = (W + 31) / 32.
    
    // Recalculate tiles based on W, H
    const int tile_w = 32;
    const int tile_h = 32;
    const int tiles_x = (W + tile_w - 1) / tile_w;
    const int tiles_y = (H + tile_h - 1) / tile_h;
    const int tiles_area = tiles_x * tiles_y;
    
    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int tx = threadIdx.x + tile_x * blockDim.x;
    const int ty = threadIdx.y + tile_y * blockDim.y;
    
    const int x_base = tx * 2, y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    const int x2 = tx, y2 = ty;
    const int x4 = x2 / 2, y4 = y2 / 2;
    const int q2x = x2 % 2, q2y = y2 % 2;
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = EVEN_DIMS ? (y4 * W4 + x4) : (min(y4, H4-1) * W4 + min(x4, W4-1));
    load_subbands_ldg(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = EVEN_DIMS ? (y2 * W2 + x2) : (min(y2, H2-1) * W2 + min(x2, W2-1));
    load_subbands_ldg(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    
    float out[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out[0][0], out[0][1], out[1][0], out[1][1]);
    
    const int out_offset = bc * H * W;
    WRITE_OUTPUT(y_base, x_base, out[0][0]);
    WRITE_OUTPUT(y_base, x_base + 1, out[0][1]);
    WRITE_OUTPUT(y_base + 1, x_base, out[1][0]);
    WRITE_OUTPUT(y_base + 1, x_base + 1, out[1][1]);
}

// =============================================================================
// 3-Level Inverse Cascade
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void ihaar2d_triple_cascade_kernel(
    const T* __restrict__ level1, const T* __restrict__ level2, const T* __restrict__ level3,
    T* __restrict__ output, int H, int W, int H2, int W2, int H4, int W4, int H8, int W8
) {
    // Grid decode
    const int tile_w = 32;
    const int tile_h = 32;
    const int tiles_x = (W + tile_w - 1) / tile_w;
    const int tiles_y = (H + tile_h - 1) / tile_h;
    const int tiles_area = tiles_x * tiles_y;
    
    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int tx = threadIdx.x + tile_x * blockDim.x;
    const int ty = threadIdx.y + tile_y * blockDim.y;
    
    const int x_base = tx * 2, y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    const int x2 = tx, y2 = ty;
    const int x4 = x2 / 2, y4 = y2 / 2;
    const int x8 = x4 / 2, y8 = y4 / 2;
    const int q2x = x2 % 2, q2y = y2 % 2;
    const int q4x = x4 % 2, q4y = y4 % 2;
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = EVEN_DIMS ? (y8 * W8 + x8) : (min(y8, H8-1) * W8 + min(x8, W8-1));
    load_subbands_ldg(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = EVEN_DIMS ? (y4 * W4 + x4) : (min(y4, H4-1) * W4 + min(x4, W4-1));
    load_subbands_ldg(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = EVEN_DIMS ? (y2 * W2 + x2) : (min(y2, H2-1) * W2 + min(x2, W2-1));
    load_subbands_ldg(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out[0][0], out[0][1], out[1][0], out[1][1]);
    
    const int out_offset = bc * H * W;
    WRITE_OUTPUT(y_base, x_base, out[0][0]);
    WRITE_OUTPUT(y_base, x_base + 1, out[0][1]);
    WRITE_OUTPUT(y_base + 1, x_base, out[1][0]);
    WRITE_OUTPUT(y_base + 1, x_base + 1, out[1][1]);
}

// =============================================================================
// 4-Level Inverse Cascade
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void ihaar2d_quad_cascade_kernel(
    const T* __restrict__ level1, const T* __restrict__ level2,
    const T* __restrict__ level3, const T* __restrict__ level4,
    T* __restrict__ output, int H, int W, int H2, int W2, int H4, int W4, int H8, int W8, int H16, int W16
) {
    // Grid decode
    const int tile_w = 32;
    const int tile_h = 32;
    const int tiles_x = (W + tile_w - 1) / tile_w;
    const int tiles_y = (H + tile_h - 1) / tile_h;
    const int tiles_area = tiles_x * tiles_y;
    
    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int tx = threadIdx.x + tile_x * blockDim.x;
    const int ty = threadIdx.y + tile_y * blockDim.y;
    
    const int x_base = tx * 2, y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    const int x2 = tx, y2 = ty;
    const int x4 = x2 / 2, y4 = y2 / 2;
    const int x8 = x4 / 2, y8 = y4 / 2;
    const int x16 = x8 / 2, y16 = y8 / 2;
    const int q2x = x2 % 2, q2y = y2 % 2;
    const int q4x = x4 % 2, q4y = y4 % 2;
    const int q8x = x8 % 2, q8y = y8 % 2;
    
    // Level 4
    float l4_ll, l4_lh, l4_hl, l4_hh;
    int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
    int idx4 = EVEN_DIMS ? (y16 * W16 + x16) : (min(y16, H16-1) * W16 + min(x16, W16-1));
    load_subbands_ldg(level4, offset4, plane4, idx4, l4_ll, l4_lh, l4_hl, l4_hh);
    float r4[2][2];
    ihaar_step(l4_ll, l4_lh, l4_hl, l4_hh, r4[0][0], r4[0][1], r4[1][0], r4[1][1]);
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = EVEN_DIMS ? (y8 * W8 + x8) : (min(y8, H8-1) * W8 + min(x8, W8-1));
    load_subbands_ldg(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    l3_ll += r4[q8y][q8x];
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = EVEN_DIMS ? (y4 * W4 + x4) : (min(y4, H4-1) * W4 + min(x4, W4-1));
    load_subbands_ldg(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = EVEN_DIMS ? (y2 * W2 + x2) : (min(y2, H2-1) * W2 + min(x2, W2-1));
    load_subbands_ldg(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out[0][0], out[0][1], out[1][0], out[1][1]);
    
    const int out_offset = bc * H * W;
    WRITE_OUTPUT(y_base, x_base, out[0][0]);
    WRITE_OUTPUT(y_base, x_base + 1, out[0][1]);
    WRITE_OUTPUT(y_base + 1, x_base, out[1][0]);
    WRITE_OUTPUT(y_base + 1, x_base + 1, out[1][1]);
}

// =============================================================================
// 5-Level Inverse Cascade
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void ihaar2d_quint_cascade_kernel(
    const T* __restrict__ level1, const T* __restrict__ level2,
    const T* __restrict__ level3, const T* __restrict__ level4, const T* __restrict__ level5,
    T* __restrict__ output, int H, int W, int H2, int W2, int H4, int W4, int H8, int W8, int H16, int W16, int H32, int W32
) {
    // Grid decode
    const int tile_w = 32;
    const int tile_h = 32;
    const int tiles_x = (W + tile_w - 1) / tile_w;
    const int tiles_y = (H + tile_h - 1) / tile_h;
    const int tiles_area = tiles_x * tiles_y;
    
    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int tx = threadIdx.x + tile_x * blockDim.x;
    const int ty = threadIdx.y + tile_y * blockDim.y;
    
    const int x_base = tx * 2, y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    const int x2 = tx, y2 = ty;
    const int x4 = x2 / 2, y4 = y2 / 2;
    const int x8 = x4 / 2, y8 = y4 / 2;
    const int x16 = x8 / 2, y16 = y8 / 2;
    const int x32 = x16 / 2, y32 = y16 / 2;
    const int q2x = x2 % 2, q2y = y2 % 2;
    const int q4x = x4 % 2, q4y = y4 % 2;
    const int q8x = x8 % 2, q8y = y8 % 2;
    const int q16x = x16 % 2, q16y = y16 % 2;
    
    // Level 5
    float l5_ll, l5_lh, l5_hl, l5_hh;
    int plane5 = H32 * W32, offset5 = bc * 4 * plane5;
    int idx5 = EVEN_DIMS ? (y32 * W32 + x32) : (min(y32, H32-1) * W32 + min(x32, W32-1));
    load_subbands_ldg(level5, offset5, plane5, idx5, l5_ll, l5_lh, l5_hl, l5_hh);
    float r5[2][2];
    ihaar_step(l5_ll, l5_lh, l5_hl, l5_hh, r5[0][0], r5[0][1], r5[1][0], r5[1][1]);
    
    // Level 4
    float l4_ll, l4_lh, l4_hl, l4_hh;
    int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
    int idx4 = EVEN_DIMS ? (y16 * W16 + x16) : (min(y16, H16-1) * W16 + min(x16, W16-1));
    load_subbands_ldg(level4, offset4, plane4, idx4, l4_ll, l4_lh, l4_hl, l4_hh);
    l4_ll += r5[q16y][q16x];
    float r4[2][2];
    ihaar_step(l4_ll, l4_lh, l4_hl, l4_hh, r4[0][0], r4[0][1], r4[1][0], r4[1][1]);
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = EVEN_DIMS ? (y8 * W8 + x8) : (min(y8, H8-1) * W8 + min(x8, W8-1));
    load_subbands_ldg(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    l3_ll += r4[q8y][q8x];
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = EVEN_DIMS ? (y4 * W4 + x4) : (min(y4, H4-1) * W4 + min(x4, W4-1));
    load_subbands_ldg(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = EVEN_DIMS ? (y2 * W2 + x2) : (min(y2, H2-1) * W2 + min(x2, W2-1));
    load_subbands_ldg(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out[0][0], out[0][1], out[1][0], out[1][1]);
    
    const int out_offset = bc * H * W;
    WRITE_OUTPUT(y_base, x_base, out[0][0]);
    WRITE_OUTPUT(y_base, x_base + 1, out[0][1]);
    WRITE_OUTPUT(y_base + 1, x_base, out[1][0]);
    WRITE_OUTPUT(y_base + 1, x_base + 1, out[1][1]);
}

#undef WRITE_OUTPUT

// =============================================================================
// Host Wrappers with dispatch macros
// =============================================================================

#define LAUNCH_KERNEL(kernel, grid, block, T, even, ...) \
    if (even) { kernel<T, true><<<grid, block>>>(__VA_ARGS__); } \
    else { kernel<T, false><<<grid, block>>>(__VA_ARGS__); }

#define DISPATCH_DTYPE(kernel, grid, block, dtype, even, ...) \
    if (dtype == torch::kFloat32) { LAUNCH_KERNEL(kernel, grid, block, float, even, __VA_ARGS__) } \
    else if (dtype == torch::kFloat16) { LAUNCH_KERNEL(kernel, grid, block, __half, even, __VA_ARGS__) } \
    else if (dtype == torch::kBFloat16) { LAUNCH_KERNEL(kernel, grid, block, __nv_bfloat16, even, __VA_ARGS__) } \
    else { TORCH_CHECK(false, "Unsupported dtype"); }

void ihaar2d_double_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor output) {
    int B = level1.size(0), C = level1.size(1);
    int H = output.size(2), W = output.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    
    dim3 block(16, 16);
    // Tile size is 32 (16 threads * 2 pixels/thread)
    int tile_size = 32;
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    bool even = (H % 4 == 0) && (W % 4 == 0);
    
    auto dtype = level1.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_KERNEL(ihaar2d_double_cascade_kernel, grid, block, float, even,
            level1.data_ptr<float>(), level2.data_ptr<float>(),
            output.data_ptr<float>(), H, W, H2, W2, H4, W4)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_KERNEL(ihaar2d_double_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), H, W, H2, W2, H4, W4)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_KERNEL(ihaar2d_double_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), H, W, H2, W2, H4, W4)
    }
}

void ihaar2d_triple_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor output) {
    int B = level1.size(0), C = level1.size(1);
    int H = output.size(2), W = output.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    int H8 = level3.size(3), W8 = level3.size(4);
    
    dim3 block(16, 16);
    // Tile size is 32 (16 threads * 2 pixels/thread)
    int tile_size = 32;
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    bool even = (H % 8 == 0) && (W % 8 == 0);
    
    auto dtype = level1.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_KERNEL(ihaar2d_triple_cascade_kernel, grid, block, float, even,
            level1.data_ptr<float>(), level2.data_ptr<float>(), level3.data_ptr<float>(),
            output.data_ptr<float>(), H, W, H2, W2, H4, W4, H8, W8)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_KERNEL(ihaar2d_triple_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level3.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), H, W, H2, W2, H4, W4, H8, W8)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_KERNEL(ihaar2d_triple_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level3.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), H, W, H2, W2, H4, W4, H8, W8)
    }
}

void ihaar2d_quad_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor output) {
    int B = level1.size(0), C = level1.size(1);
    int H = output.size(2), W = output.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    int H8 = level3.size(3), W8 = level3.size(4);
    int H16 = level4.size(3), W16 = level4.size(4);
    
    dim3 block(16, 16);
    // Tile size is 32 (16 threads * 2 pixels/thread)
    int tile_size = 32;
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    bool even = (H % 16 == 0) && (W % 16 == 0);
    
    auto dtype = level1.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_KERNEL(ihaar2d_quad_cascade_kernel, grid, block, float, even,
            level1.data_ptr<float>(), level2.data_ptr<float>(), level3.data_ptr<float>(), level4.data_ptr<float>(),
            output.data_ptr<float>(), H, W, H2, W2, H4, W4, H8, W8, H16, W16)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_KERNEL(ihaar2d_quad_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level3.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level4.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), H, W, H2, W2, H4, W4, H8, W8, H16, W16)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_KERNEL(ihaar2d_quad_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level3.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level4.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), H, W, H2, W2, H4, W4, H8, W8, H16, W16)
    }
}

void ihaar2d_quint_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor level5, torch::Tensor output) {
    int B = level1.size(0), C = level1.size(1);
    int H = output.size(2), W = output.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    int H8 = level3.size(3), W8 = level3.size(4);
    int H16 = level4.size(3), W16 = level4.size(4);
    int H32 = level5.size(3), W32 = level5.size(4);
    
    dim3 block(16, 16);
    // Tile size is 32 (16 threads * 2 pixels/thread)
    int tile_size = 32;
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    bool even = (H % 32 == 0) && (W % 32 == 0);
    
    auto dtype = level1.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_KERNEL(ihaar2d_quint_cascade_kernel, grid, block, float, even,
            level1.data_ptr<float>(), level2.data_ptr<float>(), level3.data_ptr<float>(), level4.data_ptr<float>(), level5.data_ptr<float>(),
            output.data_ptr<float>(), H, W, H2, W2, H4, W4, H8, W8, H16, W16, H32, W32)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_KERNEL(ihaar2d_quint_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level3.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level4.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(level5.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), H, W, H2, W2, H4, W4, H8, W8, H16, W16, H32, W32)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_KERNEL(ihaar2d_quint_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level3.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level4.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(level5.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), H, W, H2, W2, H4, W4, H8, W8, H16, W16, H32, W32)
    }
}
