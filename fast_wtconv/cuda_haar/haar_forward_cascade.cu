#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// Optimized Forward Haar 2D Cascade - V2 (All Levels: 2, 3, 4, 5)
//
// Single-kernel cascade that computes all levels at once with:
// 1. __ldg() for texture cache loads
// 2. 2x2 input per thread (each thread reads 2x2 block, contributes to all levels)
// 3. Shared memory for inter-level data
// 4. Fast path template for even dimensions
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

// Forward Haar step
__device__ __forceinline__ void haar_step(
    float a, float b, float c, float d,
    float& ll, float& lh, float& hl, float& hh
) {
    float sum_ac = a + c, sum_bd = b + d;
    float diff_ac = a - c, diff_bd = b - d;
    ll = 0.5f * (sum_ac + sum_bd);
    lh = 0.5f * (diff_ac + diff_bd);
    hl = 0.5f * (sum_ac - sum_bd);
    hh = 0.5f * (diff_ac - diff_bd);
}

// Helper to write 4 subbands at once
template<typename T>
__device__ __forceinline__ void write_subbands(
    T* __restrict__ output, int offset, int plane, int idx,
    float ll, float lh, float hl, float hh
) {
    output[offset + 0 * plane + idx] = from_float<T>(ll);
    output[offset + 1 * plane + idx] = from_float<T>(lh);
    output[offset + 2 * plane + idx] = from_float<T>(hl);
    output[offset + 3 * plane + idx] = from_float<T>(hh);
}

// =============================================================================
// 2-Level Forward Cascade
// Each thread reads 4x4 input block, outputs 2x2 level1 + 1x1 level2
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void haar2d_double_cascade_kernel(
    const T* __restrict__ input, T* __restrict__ level1, T* __restrict__ level2,
    int H, int W, int H2, int W2, int H4, int W4
) {
    const int tiles_x = (W4 + 15) / 16;
    const int tiles_y = (H4 + 15) / 16;
    const int tiles_area = tiles_x * tiles_y;
    
    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int tx = threadIdx.x + tile_x * blockDim.x;
    const int ty = threadIdx.y + tile_y * blockDim.y;
    
    // Each thread processes a 4x4 input block -> 2x2 level1 -> 1x1 level2
    const int x_base = tx * 4, y_base = ty * 4;
    if (y_base >= H || x_base >= W) return;
    
    const int in_offset = bc * H * W;
    
    // Read 4x4 input using __ldg
    float in[4][4];
    #pragma unroll
    for (int dy = 0; dy < 4; dy++) {
        #pragma unroll
        for (int dx = 0; dx < 4; dx++) {
            int y = EVEN_DIMS ? (y_base + dy) : min(y_base + dy, H - 1);
            int x = EVEN_DIMS ? (x_base + dx) : min(x_base + dx, W - 1);
            in[dy][dx] = to_float(__ldg(&input[in_offset + y * W + x]));
        }
    }
    
    // Level 1: 4x4 -> 2x2 (compute 4 Haar transforms)
    float l1[4][2][2];  // [subband][y][x]
    #pragma unroll
    for (int qy = 0; qy < 2; qy++) {
        #pragma unroll
        for (int qx = 0; qx < 2; qx++) {
            float a = in[qy*2][qx*2], b = in[qy*2][qx*2+1];
            float c = in[qy*2+1][qx*2], d = in[qy*2+1][qx*2+1];
            haar_step(a, b, c, d, l1[0][qy][qx], l1[1][qy][qx], l1[2][qy][qx], l1[3][qy][qx]);
        }
    }
    
    // Write level 1 output (2x2 per thread = 4 output pixels)
    int plane1 = H2 * W2;
    int offset1 = bc * 4 * plane1;
    int out1_y = ty * 2, out1_x = tx * 2;
    
    #pragma unroll
    for (int dy = 0; dy < 2; dy++) {
        #pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            if (EVEN_DIMS || (out1_y + dy < H2 && out1_x + dx < W2)) {
                int idx = (out1_y + dy) * W2 + (out1_x + dx);
                write_subbands(level1, offset1, plane1, idx, 
                              l1[0][dy][dx], l1[1][dy][dx], l1[2][dy][dx], l1[3][dy][dx]);
            }
        }
    }
    
    // Level 2: Apply Haar to 2x2 LL subband -> 1x1 level2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    haar_step(l1[0][0][0], l1[0][0][1], l1[0][1][0], l1[0][1][1], l2_ll, l2_lh, l2_hl, l2_hh);
    
    // Write level 2 output (1 per thread)
    int plane2 = H4 * W4;
    int offset2 = bc * 4 * plane2;
    int out2_y = ty, out2_x = tx;
    
    if (EVEN_DIMS || (out2_y < H4 && out2_x < W4)) {
        int idx = out2_y * W4 + out2_x;
        write_subbands(level2, offset2, plane2, idx, l2_ll, l2_lh, l2_hl, l2_hh);
    }
}

// =============================================================================
// 3-Level Forward Cascade
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void haar2d_triple_cascade_kernel(
    const T* __restrict__ input, 
    T* __restrict__ level1, T* __restrict__ level2, T* __restrict__ level3,
    int H, int W, int H2, int W2, int H4, int W4, int H8, int W8
) {
    // Shared memory for level1 LL subband (needed for level2/3)
    __shared__ float smem_l1_ll[8][8];  // 8x8 LL values from 16x16 threads
    
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int tid = ty * 16 + tx;
    
    // Grid decode
    const int tiles_x = (W + 31) / 32;
    const int tiles_y = (H + 31) / 32;
    const int tiles_area = tiles_x * tiles_y;

    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;
    
    // Each block of 16x16 threads processes 32x32 input -> 16x16 level1 -> 8x8 level2 -> 4x4 level3
    const int x_base = tile_x * 32 + tx * 2;
    const int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    const int in_offset = bc * H * W;
    
    // ==== Level 1: Each thread reads 2x2 input, outputs 1x1 level1 ====
    float a, b, c, d;
    if (EVEN_DIMS || (y_base < H && x_base < W)) {
        int y0 = EVEN_DIMS ? y_base : min(y_base, H-1);
        int y1 = EVEN_DIMS ? y_base + 1 : min(y_base + 1, H-1);
        int x0 = EVEN_DIMS ? x_base : min(x_base, W-1);
        int x1 = EVEN_DIMS ? x_base + 1 : min(x_base + 1, W-1);
        
        a = to_float(__ldg(&input[in_offset + y0 * W + x0]));
        b = to_float(__ldg(&input[in_offset + y0 * W + x1]));
        c = to_float(__ldg(&input[in_offset + y1 * W + x0]));
        d = to_float(__ldg(&input[in_offset + y1 * W + x1]));
    } else {
        a = b = c = d = 0.0f;
    }
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    // Store LL to shared memory for level 2
    smem_l1_ll[ty / 2 + (ty % 2 == 0 ? 0 : 0)][tx / 2 + (tx % 2 == 0 ? 0 : 0)] = 0; // Init
    if (tx < 8 && ty < 8) {
        smem_l1_ll[ty][tx] = 0.0f;
    }
    __syncthreads();
    
    // We need to collect LL values in a 2x2 pattern
    // Thread (tx, ty) produces LL at position (tx, ty) in level1
    // For level2, we need to combine LL[2i,2j], LL[2i,2j+1], LL[2i+1,2j], LL[2i+1,2j+1]
    
    // Write level 1 output
    int plane1 = H2 * W2;
    int offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty;
    int out1_x = tile_x * 16 + tx;
    
    if (EVEN_DIMS || (out1_y < H2 && out1_x < W2)) {
        int idx = out1_y * W2 + out1_x;
        write_subbands(level1, offset1, plane1, idx, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    // Store LL to shared memory
    if (ty < 16 && tx < 16) {
        smem_l1_ll[ty % 8][tx % 8] = (tx < 8 && ty < 8) ? l1_ll : 0;
    }
    
    // Actually, simpler approach: first 64 threads store, rest wait
    __shared__ float smem_all_ll[16][16];
    smem_all_ll[ty][tx] = l1_ll;
    __syncthreads();
    
    // ==== Level 2: 16x16 -> 8x8 (first 64 threads active) ====
    float l2_ll, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_all_ll[ty*2][tx*2];
        b = smem_all_ll[ty*2][tx*2+1];
        c = smem_all_ll[ty*2+1][tx*2];
        d = smem_all_ll[ty*2+1][tx*2+1];
        
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        // Write level 2
        int plane2 = H4 * W4;
        int offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty;
        int out2_x = tile_x * 8 + tx;
        
        if (EVEN_DIMS || (out2_y < H4 && out2_x < W4)) {
            int idx = out2_y * W4 + out2_x;
            write_subbands(level2, offset2, plane2, idx, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    __syncthreads();
    
    // Store level 2 LL
    __shared__ float smem_l2_ll[8][8];
    if (tx < 8 && ty < 8) {
        smem_l2_ll[ty][tx] = l2_ll;
    }
    __syncthreads();
    
    // ==== Level 3: 8x8 -> 4x4 (first 16 threads active) ====
    if (tx < 4 && ty < 4) {
        a = smem_l2_ll[ty*2][tx*2];
        b = smem_l2_ll[ty*2][tx*2+1];
        c = smem_l2_ll[ty*2+1][tx*2];
        d = smem_l2_ll[ty*2+1][tx*2+1];
        
        float l3_ll, l3_lh, l3_hl, l3_hh;
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        // Write level 3
        int plane3 = H8 * W8;
        int offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty;
        int out3_x = tile_x * 4 + tx;
        
        if (EVEN_DIMS || (out3_y < H8 && out3_x < W8)) {
            int idx = out3_y * W8 + out3_x;
            write_subbands(level3, offset3, plane3, idx, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
}

// =============================================================================
// 4-Level Forward Cascade  
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void haar2d_quad_cascade_kernel(
    const T* __restrict__ input,
    T* __restrict__ level1, T* __restrict__ level2, T* __restrict__ level3, T* __restrict__ level4,
    int H, int W, int H2, int W2, int H4, int W4, int H8, int W8, int H16, int W16
) {
    __shared__ float smem_ll[16][16];
    
    const int tx = threadIdx.x, ty = threadIdx.y;

    // Grid decode
    const int tiles_x = (W + 31) / 32;
    const int tiles_y = (H + 31) / 32;
    const int tiles_area = tiles_x * tiles_y;

    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;
    
    const int x_base = tile_x * 32 + tx * 2;
    const int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    const int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    int y0 = EVEN_DIMS ? y_base : min(y_base, H-1);
    int y1 = EVEN_DIMS ? y_base+1 : min(y_base+1, H-1);
    int x0 = EVEN_DIMS ? x_base : min(x_base, W-1);
    int x1 = EVEN_DIMS ? x_base+1 : min(x_base+1, W-1);
    
    a = to_float(__ldg(&input[in_offset + y0 * W + x0]));
    b = to_float(__ldg(&input[in_offset + y0 * W + x1]));
    c = to_float(__ldg(&input[in_offset + y1 * W + x0]));
    d = to_float(__ldg(&input[in_offset + y1 * W + x1]));
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty, out1_x = tile_x * 16 + tx;
    if (EVEN_DIMS || (out1_y < H2 && out1_x < W2)) {
        write_subbands(level1, offset1, plane1, out1_y * W2 + out1_x, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    smem_ll[ty][tx] = l1_ll;
    __syncthreads();
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_ll[ty*2][tx*2]; b = smem_ll[ty*2][tx*2+1];
        c = smem_ll[ty*2+1][tx*2]; d = smem_ll[ty*2+1][tx*2+1];
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty, out2_x = tile_x * 8 + tx;
        if (EVEN_DIMS || (out2_y < H4 && out2_x < W4)) {
            write_subbands(level2, offset2, plane2, out2_y * W4 + out2_x, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    __syncthreads();
    
    if (tx < 8 && ty < 8) smem_ll[ty][tx] = l2_ll;
    __syncthreads();
    
    // Level 3
    float l3_ll = 0, l3_lh, l3_hl, l3_hh;
    if (tx < 4 && ty < 4) {
        a = smem_ll[ty*2][tx*2]; b = smem_ll[ty*2][tx*2+1];
        c = smem_ll[ty*2+1][tx*2]; d = smem_ll[ty*2+1][tx*2+1];
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty, out3_x = tile_x * 4 + tx;
        if (EVEN_DIMS || (out3_y < H8 && out3_x < W8)) {
            write_subbands(level3, offset3, plane3, out3_y * W8 + out3_x, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
    __syncthreads();
    
    if (tx < 4 && ty < 4) smem_ll[ty][tx] = l3_ll;
    __syncthreads();
    
    // Level 4
    if (tx < 2 && ty < 2) {
        a = smem_ll[ty*2][tx*2]; b = smem_ll[ty*2][tx*2+1];
        c = smem_ll[ty*2+1][tx*2]; d = smem_ll[ty*2+1][tx*2+1];
        float l4_ll, l4_lh, l4_hl, l4_hh;
        haar_step(a, b, c, d, l4_ll, l4_lh, l4_hl, l4_hh);
        
        int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
        int out4_y = tile_y * 2 + ty, out4_x = tile_x * 2 + tx;
        if (EVEN_DIMS || (out4_y < H16 && out4_x < W16)) {
            write_subbands(level4, offset4, plane4, out4_y * W16 + out4_x, l4_ll, l4_lh, l4_hl, l4_hh);
        }
    }
}

// =============================================================================
// 5-Level Forward Cascade
// =============================================================================
template<typename T, bool EVEN_DIMS>
__global__ void haar2d_quint_cascade_kernel(
    const T* __restrict__ input,
    T* __restrict__ level1, T* __restrict__ level2, T* __restrict__ level3, 
    T* __restrict__ level4, T* __restrict__ level5,
    int H, int W, int H2, int W2, int H4, int W4, int H8, int W8, int H16, int W16, int H32, int W32
) {
    __shared__ float smem_ll[16][16];
    
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    
    // Grid decode
    const int tiles_x = (W + 31) / 32;
    const int tiles_y = (H + 31) / 32;
    const int tiles_area = tiles_x * tiles_y;

    if (tiles_area == 0) return;

    const int bc = blockIdx.x / tiles_area;
    const int tile_idx = blockIdx.x % tiles_area;
    const int tile_y = tile_idx / tiles_x;
    const int tile_x = tile_idx % tiles_x;

    const int x_base = tile_x * 32 + tx * 2;
    const int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    const int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    int y0 = EVEN_DIMS ? y_base : min(y_base, H-1);
    int y1 = EVEN_DIMS ? y_base+1 : min(y_base+1, H-1);
    int x0 = EVEN_DIMS ? x_base : min(x_base, W-1);
    int x1 = EVEN_DIMS ? x_base+1 : min(x_base+1, W-1);
    
    a = to_float(__ldg(&input[in_offset + y0 * W + x0]));
    b = to_float(__ldg(&input[in_offset + y0 * W + x1]));
    c = to_float(__ldg(&input[in_offset + y1 * W + x0]));
    d = to_float(__ldg(&input[in_offset + y1 * W + x1]));
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty, out1_x = tile_x * 16 + tx;
    if (EVEN_DIMS || (out1_y < H2 && out1_x < W2)) {
        write_subbands(level1, offset1, plane1, out1_y * W2 + out1_x, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    smem_ll[ty][tx] = l1_ll;
    __syncthreads();
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_ll[ty*2][tx*2]; b = smem_ll[ty*2][tx*2+1];
        c = smem_ll[ty*2+1][tx*2]; d = smem_ll[ty*2+1][tx*2+1];
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty, out2_x = tile_x * 8 + tx;
        if (EVEN_DIMS || (out2_y < H4 && out2_x < W4)) {
            write_subbands(level2, offset2, plane2, out2_y * W4 + out2_x, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    __syncthreads();
    
    if (tx < 8 && ty < 8) smem_ll[ty][tx] = l2_ll;
    __syncthreads();
    
    // Level 3
    float l3_ll = 0, l3_lh, l3_hl, l3_hh;
    if (tx < 4 && ty < 4) {
        a = smem_ll[ty*2][tx*2]; b = smem_ll[ty*2][tx*2+1];
        c = smem_ll[ty*2+1][tx*2]; d = smem_ll[ty*2+1][tx*2+1];
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty, out3_x = tile_x * 4 + tx;
        if (EVEN_DIMS || (out3_y < H8 && out3_x < W8)) {
            write_subbands(level3, offset3, plane3, out3_y * W8 + out3_x, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
    __syncthreads();
    
    if (tx < 4 && ty < 4) smem_ll[ty][tx] = l3_ll;
    __syncthreads();
    
    // Level 4
    float l4_ll = 0, l4_lh, l4_hl, l4_hh;
    if (tx < 2 && ty < 2) {
        a = smem_ll[ty*2][tx*2]; b = smem_ll[ty*2][tx*2+1];
        c = smem_ll[ty*2+1][tx*2]; d = smem_ll[ty*2+1][tx*2+1];
        haar_step(a, b, c, d, l4_ll, l4_lh, l4_hl, l4_hh);
        
        int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
        int out4_y = tile_y * 2 + ty, out4_x = tile_x * 2 + tx;
        if (EVEN_DIMS || (out4_y < H16 && out4_x < W16)) {
            write_subbands(level4, offset4, plane4, out4_y * W16 + out4_x, l4_ll, l4_lh, l4_hl, l4_hh);
        }
    }
    __syncthreads();
    
    if (tx < 2 && ty < 2) smem_ll[ty][tx] = l4_ll;
    __syncthreads();
    
    // Level 5 (single thread)
    if (tid == 0) {
        a = smem_ll[0][0]; b = smem_ll[0][1];
        c = smem_ll[1][0]; d = smem_ll[1][1];
        float l5_ll, l5_lh, l5_hl, l5_hh;
        haar_step(a, b, c, d, l5_ll, l5_lh, l5_hl, l5_hh);
        
        int plane5 = H32 * W32, offset5 = bc * 4 * plane5;
        int out5_idx = tile_y * W32 + tile_x;
        if (EVEN_DIMS || out5_idx < plane5) {
            write_subbands(level5, offset5, plane5, out5_idx, l5_ll, l5_lh, l5_hl, l5_hh);
        }
    }
}

// =============================================================================
// Host Wrappers
// =============================================================================

#define LAUNCH_FWD_KERNEL(kernel, grid, block, T, even, ...) \
    if (even) { kernel<T, true><<<grid, block>>>(__VA_ARGS__); } \
    else { kernel<T, false><<<grid, block>>>(__VA_ARGS__); }

void haar2d_double_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2) {
    int B = input.size(0), C = input.size(1);
    int H = input.size(2), W = input.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    
    // Each thread processes 4x4 input
    dim3 block(16, 16);
    // Linearize grid to avoid Z-limit overflow (65535) with large B*C
    int tiles_x = (W4 + 15) / 16;
    int tiles_y = (H4 + 15) / 16;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    bool even = (H % 4 == 0) && (W % 4 == 0);
    
    auto dtype = input.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_FWD_KERNEL(haar2d_double_cascade_kernel, grid, block, float, even,
            input.data_ptr<float>(), level1.data_ptr<float>(), level2.data_ptr<float>(),
            H, W, H2, W2, H4, W4)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_double_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level2.data_ptr<at::Half>()),
            H, W, H2, W2, H4, W4)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_double_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            H, W, H2, W2, H4, W4)
    }
}

void haar2d_triple_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2, torch::Tensor level3) {
    int B = input.size(0), C = input.size(1);
    int H = input.size(2), W = input.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    int H8 = level3.size(3), W8 = level3.size(4);
    
    dim3 block(16, 16);
    // Tiling logic: one block per 32x32 input tile
    int tiles_x = (W + 31) / 32;
    int tiles_y = (H + 31) / 32;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);

    bool even = (H % 8 == 0) && (W % 8 == 0);
    
    auto dtype = input.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_FWD_KERNEL(haar2d_triple_cascade_kernel, grid, block, float, even,
            input.data_ptr<float>(), level1.data_ptr<float>(), level2.data_ptr<float>(), level3.data_ptr<float>(),
            H, W, H2, W2, H4, W4, H8, W8)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_triple_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level3.data_ptr<at::Half>()),
            H, W, H2, W2, H4, W4, H8, W8)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_triple_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level3.data_ptr<at::BFloat16>()),
            H, W, H2, W2, H4, W4, H8, W8)
    }
}

void haar2d_quad_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2, 
                             torch::Tensor level3, torch::Tensor level4) {
    int B = input.size(0), C = input.size(1);
    int H = input.size(2), W = input.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    int H8 = level3.size(3), W8 = level3.size(4);
    int H16 = level4.size(3), W16 = level4.size(4);
    
    dim3 block(16, 16);
    // Tiling logic: same as triple, one block per 32x32 input tile
    int tiles_x = (W + 31) / 32;
    int tiles_y = (H + 31) / 32;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    bool even = (H % 16 == 0) && (W % 16 == 0);
    
    auto dtype = input.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_FWD_KERNEL(haar2d_quad_cascade_kernel, grid, block, float, even,
            input.data_ptr<float>(), level1.data_ptr<float>(), level2.data_ptr<float>(), 
            level3.data_ptr<float>(), level4.data_ptr<float>(),
            H, W, H2, W2, H4, W4, H8, W8, H16, W16)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_quad_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level3.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level4.data_ptr<at::Half>()),
            H, W, H2, W2, H4, W4, H8, W8, H16, W16)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_quad_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level3.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level4.data_ptr<at::BFloat16>()),
            H, W, H2, W2, H4, W4, H8, W8, H16, W16)
    }
}

void haar2d_quint_cascade(torch::Tensor input, torch::Tensor level1, torch::Tensor level2,
                              torch::Tensor level3, torch::Tensor level4, torch::Tensor level5) {
    int B = input.size(0), C = input.size(1);
    int H = input.size(2), W = input.size(3);
    int H2 = level1.size(3), W2 = level1.size(4);
    int H4 = level2.size(3), W4 = level2.size(4);
    int H8 = level3.size(3), W8 = level3.size(4);
    int H16 = level4.size(3), W16 = level4.size(4);
    int H32 = level5.size(3), W32 = level5.size(4);
    
    dim3 block(16, 16);
    // Tiling logic: same as triple, one block per 32x32 input tile
    int tiles_x = (W + 31) / 32;
    int tiles_y = (H + 31) / 32;
    long long total_tiles = (long long)tiles_x * tiles_y * B * C;
    dim3 grid(total_tiles, 1, 1);
    
    bool even = (H % 32 == 0) && (W % 32 == 0);
    
    auto dtype = input.dtype();
    if (dtype == torch::kFloat32) {
        LAUNCH_FWD_KERNEL(haar2d_quint_cascade_kernel, grid, block, float, even,
            input.data_ptr<float>(), level1.data_ptr<float>(), level2.data_ptr<float>(),
            level3.data_ptr<float>(), level4.data_ptr<float>(), level5.data_ptr<float>(),
            H, W, H2, W2, H4, W4, H8, W8, H16, W16, H32, W32)
    } else if (dtype == torch::kFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_quint_cascade_kernel, grid, block, __half, even,
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level1.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level2.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level3.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level4.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(level5.data_ptr<at::Half>()),
            H, W, H2, W2, H4, W4, H8, W8, H16, W16, H32, W32)
    } else if (dtype == torch::kBFloat16) {
        LAUNCH_FWD_KERNEL(haar2d_quint_cascade_kernel, grid, block, __nv_bfloat16, even,
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level1.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level3.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level4.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(level5.data_ptr<at::BFloat16>()),
            H, W, H2, W2, H4, W4, H8, W8, H16, W16, H32, W32)
    }
}

// =============================================================================
// BACKWARD KERNELS
// The backward of forward Haar cascade is the inverse Haar cascade.
// We call the ihaar2d_*_cascade functions from haar_inverse_cascade.cu
// =============================================================================

// Declare external inverse cascade functions (from haar_inverse_cascade.cu)
extern void ihaar2d_double_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor output);
extern void ihaar2d_triple_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor output);
extern void ihaar2d_quad_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor output);
extern void ihaar2d_quint_cascade(torch::Tensor level1, torch::Tensor level2, torch::Tensor level3, torch::Tensor level4, torch::Tensor level5, torch::Tensor output);

void haar2d_double_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, torch::Tensor grad_input) {
    // Backward of forward Haar is inverse Haar
    ihaar2d_double_cascade(grad_level1, grad_level2, grad_input);
}

void haar2d_triple_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, 
                                        torch::Tensor grad_level3, torch::Tensor grad_input) {
    ihaar2d_triple_cascade(grad_level1, grad_level2, grad_level3, grad_input);
}

void haar2d_quad_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, 
                                      torch::Tensor grad_level3, torch::Tensor grad_level4, 
                                      torch::Tensor grad_input) {
    ihaar2d_quad_cascade(grad_level1, grad_level2, grad_level3, grad_level4, grad_input);
}

void haar2d_quint_cascade_backward(torch::Tensor grad_level1, torch::Tensor grad_level2, 
                                       torch::Tensor grad_level3, torch::Tensor grad_level4,
                                       torch::Tensor grad_level5, torch::Tensor grad_input) {
    ihaar2d_quint_cascade(grad_level1, grad_level2, grad_level3, grad_level4, grad_level5, grad_input);
}
