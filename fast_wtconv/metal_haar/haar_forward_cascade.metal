#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Optimized Forward Haar 2D Cascade - Metal (All Levels: 2, 3, 4, 5)
//
// Single-kernel cascade that computes all levels at once with:
// 1. Threadgroup (shared) memory for inter-level data
// 2. Optimized memory access patterns
// =============================================================================

// Forward Haar step helper
inline void haar_step(float a, float b, float c, float d,
                      thread float& ll, thread float& lh, thread float& hl, thread float& hh) {
    float sum_ac = a + c;
    float sum_bd = b + d;
    float diff_ac = a - c;
    float diff_bd = b - d;
    ll = 0.5f * (sum_ac + sum_bd);
    lh = 0.5f * (diff_ac + diff_bd);
    hl = 0.5f * (sum_ac - sum_bd);
    hh = 0.5f * (diff_ac - diff_bd);
}

// Helper to write 4 subbands at once
inline void write_subbands(device float* output, int offset, int plane, int idx,
                           float ll, float lh, float hl, float hh) {
    output[offset + 0 * plane + idx] = ll;
    output[offset + 1 * plane + idx] = lh;
    output[offset + 2 * plane + idx] = hl;
    output[offset + 3 * plane + idx] = hh;
}

// Half precision helpers
inline void write_subbands_half(device half* output, int offset, int plane, int idx,
                                float ll, float lh, float hl, float hh) {
    output[offset + 0 * plane + idx] = half(ll);
    output[offset + 1 * plane + idx] = half(lh);
    output[offset + 2 * plane + idx] = half(hl);
    output[offset + 3 * plane + idx] = half(hh);
}

// =============================================================================
// 2-Level Forward Cascade
// Each thread reads 4x4 input block, outputs 2x2 level1 + 1x1 level2
// =============================================================================
kernel void haar2d_double_cascade_kernel(
    device const float* input [[buffer(0)]],
    device float* level1 [[buffer(1)]],
    device float* level2 [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    constant int& H2 [[buffer(5)]],
    constant int& W2 [[buffer(6)]],
    constant int& H4 [[buffer(7)]],
    constant int& W4 [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    // Each thread processes a 4x4 input block -> 2x2 level1 -> 1x1 level2
    int x_base = tx * 4;
    int y_base = ty * 4;
    if (y_base >= H || x_base >= W) return;
    
    int in_offset = bc * H * W;
    
    // Read 4x4 input
    float in_vals[4][4];
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            int y = min(y_base + dy, H - 1);
            int x = min(x_base + dx, W - 1);
            in_vals[dy][dx] = input[in_offset + y * W + x];
        }
    }
    
    // Level 1: 4x4 -> 2x2 (compute 4 Haar transforms)
    float l1[4][2][2];  // [subband][y][x]
    for (int qy = 0; qy < 2; qy++) {
        for (int qx = 0; qx < 2; qx++) {
            float a = in_vals[qy*2][qx*2];
            float b = in_vals[qy*2][qx*2+1];
            float c = in_vals[qy*2+1][qx*2];
            float d = in_vals[qy*2+1][qx*2+1];
            haar_step(a, b, c, d, l1[0][qy][qx], l1[1][qy][qx], l1[2][qy][qx], l1[3][qy][qx]);
        }
    }
    
    // Write level 1 output (2x2 per thread = 4 output pixels)
    int plane1 = H2 * W2;
    int offset1 = bc * 4 * plane1;
    int out1_y = ty * 2;
    int out1_x = tx * 2;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            if (out1_y + dy < H2 && out1_x + dx < W2) {
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
    int out2_y = ty;
    int out2_x = tx;
    
    if (out2_y < H4 && out2_x < W4) {
        int idx = out2_y * W4 + out2_x;
        write_subbands(level2, offset2, plane2, idx, l2_ll, l2_lh, l2_hl, l2_hh);
    }
}

// =============================================================================
// 3-Level Forward Cascade
// =============================================================================
kernel void haar2d_triple_cascade_kernel(
    device const float* input [[buffer(0)]],
    device float* level1 [[buffer(1)]],
    device float* level2 [[buffer(2)]],
    device float* level3 [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& H2 [[buffer(6)]],
    constant int& W2 [[buffer(7)]],
    constant int& H4 [[buffer(8)]],
    constant int& W4 [[buffer(9)]],
    constant int& H8 [[buffer(10)]],
    constant int& W8 [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
) {
    int tx = tid.x;  // 0-15
    int ty = tid.y;  // 0-15
    int bc = gid.z;
    
    // Each block of 16x16 threads processes 32x32 input -> 16x16 level1 -> 8x8 level2 -> 4x4 level3
    int tile_x = gid.x / 16;
    int tile_y = gid.y / 16;
    int x_base = tile_x * 32 + tx * 2;
    int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    int in_offset = bc * H * W;
    
    // ==== Level 1: Each thread reads 2x2 input, outputs 1x1 level1 ====
    float a, b, c, d;
    if (y_base < H && x_base < W) {
        int y0 = min(y_base, H-1);
        int y1 = min(y_base + 1, H-1);
        int x0 = min(x_base, W-1);
        int x1 = min(x_base + 1, W-1);
        
        a = input[in_offset + y0 * W + x0];
        b = input[in_offset + y0 * W + x1];
        c = input[in_offset + y1 * W + x0];
        d = input[in_offset + y1 * W + x1];
    } else {
        a = b = c = d = 0.0f;
    }
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    // Write level 1 output
    int plane1 = H2 * W2;
    int offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty;
    int out1_x = tile_x * 16 + tx;
    
    if (out1_y < H2 && out1_x < W2) {
        int idx = out1_y * W2 + out1_x;
        write_subbands(level1, offset1, plane1, idx, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    // Store LL to shared memory for level 2
    threadgroup float* smem_all_ll = smem;
    smem_all_ll[ty * 16 + tx] = l1_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ==== Level 2: 16x16 -> 8x8 (first 64 threads active) ====
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_all_ll[(ty*2) * 16 + (tx*2)];
        b = smem_all_ll[(ty*2) * 16 + (tx*2+1)];
        c = smem_all_ll[(ty*2+1) * 16 + (tx*2)];
        d = smem_all_ll[(ty*2+1) * 16 + (tx*2+1)];
        
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4;
        int offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty;
        int out2_x = tile_x * 8 + tx;
        
        if (out2_y < H4 && out2_x < W4) {
            int idx = out2_y * W4 + out2_x;
            write_subbands(level2, offset2, plane2, idx, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Store level 2 LL
    threadgroup float* smem_l2_ll = smem;
    if (tx < 8 && ty < 8) {
        smem_l2_ll[ty * 8 + tx] = l2_ll;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ==== Level 3: 8x8 -> 4x4 (first 16 threads active) ====
    if (tx < 4 && ty < 4) {
        a = smem_l2_ll[(ty*2) * 8 + (tx*2)];
        b = smem_l2_ll[(ty*2) * 8 + (tx*2+1)];
        c = smem_l2_ll[(ty*2+1) * 8 + (tx*2)];
        d = smem_l2_ll[(ty*2+1) * 8 + (tx*2+1)];
        
        float l3_ll, l3_lh, l3_hl, l3_hh;
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8;
        int offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty;
        int out3_x = tile_x * 4 + tx;
        
        if (out3_y < H8 && out3_x < W8) {
            int idx = out3_y * W8 + out3_x;
            write_subbands(level3, offset3, plane3, idx, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
}

// =============================================================================
// 4-Level Forward Cascade
// =============================================================================
kernel void haar2d_quad_cascade_kernel(
    device const float* input [[buffer(0)]],
    device float* level1 [[buffer(1)]],
    device float* level2 [[buffer(2)]],
    device float* level3 [[buffer(3)]],
    device float* level4 [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& W [[buffer(6)]],
    constant int& H2 [[buffer(7)]],
    constant int& W2 [[buffer(8)]],
    constant int& H4 [[buffer(9)]],
    constant int& W4 [[buffer(10)]],
    constant int& H8 [[buffer(11)]],
    constant int& W8 [[buffer(12)]],
    constant int& H16 [[buffer(13)]],
    constant int& W16 [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
) {
    int tx = tid.x;
    int ty = tid.y;
    int bc = gid.z;
    int tile_x = gid.x / 16;
    int tile_y = gid.y / 16;
    int x_base = tile_x * 32 + tx * 2;
    int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    int y0 = min(y_base, H-1);
    int y1 = min(y_base+1, H-1);
    int x0 = min(x_base, W-1);
    int x1 = min(x_base+1, W-1);
    
    a = input[in_offset + y0 * W + x0];
    b = input[in_offset + y0 * W + x1];
    c = input[in_offset + y1 * W + x0];
    d = input[in_offset + y1 * W + x1];
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty, out1_x = tile_x * 16 + tx;
    if (out1_y < H2 && out1_x < W2) {
        write_subbands(level1, offset1, plane1, out1_y * W2 + out1_x, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    threadgroup float* smem_ll = smem;
    smem_ll[ty * 16 + tx] = l1_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_ll[(ty*2) * 16 + (tx*2)];
        b = smem_ll[(ty*2) * 16 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 16 + (tx*2)];
        d = smem_ll[(ty*2+1) * 16 + (tx*2+1)];
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty, out2_x = tile_x * 8 + tx;
        if (out2_y < H4 && out2_x < W4) {
            write_subbands(level2, offset2, plane2, out2_y * W4 + out2_x, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 8 && ty < 8) smem_ll[ty * 8 + tx] = l2_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 3
    float l3_ll = 0, l3_lh, l3_hl, l3_hh;
    if (tx < 4 && ty < 4) {
        a = smem_ll[(ty*2) * 8 + (tx*2)];
        b = smem_ll[(ty*2) * 8 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 8 + (tx*2)];
        d = smem_ll[(ty*2+1) * 8 + (tx*2+1)];
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty, out3_x = tile_x * 4 + tx;
        if (out3_y < H8 && out3_x < W8) {
            write_subbands(level3, offset3, plane3, out3_y * W8 + out3_x, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 4 && ty < 4) smem_ll[ty * 4 + tx] = l3_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 4
    if (tx < 2 && ty < 2) {
        a = smem_ll[(ty*2) * 4 + (tx*2)];
        b = smem_ll[(ty*2) * 4 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 4 + (tx*2)];
        d = smem_ll[(ty*2+1) * 4 + (tx*2+1)];
        float l4_ll, l4_lh, l4_hl, l4_hh;
        haar_step(a, b, c, d, l4_ll, l4_lh, l4_hl, l4_hh);
        
        int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
        int out4_y = tile_y * 2 + ty, out4_x = tile_x * 2 + tx;
        if (out4_y < H16 && out4_x < W16) {
            write_subbands(level4, offset4, plane4, out4_y * W16 + out4_x, l4_ll, l4_lh, l4_hl, l4_hh);
        }
    }
}

// =============================================================================
// 5-Level Forward Cascade
// =============================================================================
kernel void haar2d_quint_cascade_kernel(
    device const float* input [[buffer(0)]],
    device float* level1 [[buffer(1)]],
    device float* level2 [[buffer(2)]],
    device float* level3 [[buffer(3)]],
    device float* level4 [[buffer(4)]],
    device float* level5 [[buffer(5)]],
    constant int& H [[buffer(6)]],
    constant int& W [[buffer(7)]],
    constant int& H2 [[buffer(8)]],
    constant int& W2 [[buffer(9)]],
    constant int& H4 [[buffer(10)]],
    constant int& W4 [[buffer(11)]],
    constant int& H8 [[buffer(12)]],
    constant int& W8 [[buffer(13)]],
    constant int& H16 [[buffer(14)]],
    constant int& W16 [[buffer(15)]],
    constant int& H32 [[buffer(16)]],
    constant int& W32 [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
) {
    int tx = tid.x;
    int ty = tid.y;
    int tid_linear = ty * 16 + tx;
    int bc = gid.z;
    int tile_x = gid.x / 16;
    int tile_y = gid.y / 16;
    int x_base = tile_x * 32 + tx * 2;
    int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    int y0 = min(y_base, H-1);
    int y1 = min(y_base+1, H-1);
    int x0 = min(x_base, W-1);
    int x1 = min(x_base+1, W-1);
    
    a = input[in_offset + y0 * W + x0];
    b = input[in_offset + y0 * W + x1];
    c = input[in_offset + y1 * W + x0];
    d = input[in_offset + y1 * W + x1];
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty, out1_x = tile_x * 16 + tx;
    if (out1_y < H2 && out1_x < W2) {
        write_subbands(level1, offset1, plane1, out1_y * W2 + out1_x, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    threadgroup float* smem_ll = smem;
    smem_ll[ty * 16 + tx] = l1_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_ll[(ty*2) * 16 + (tx*2)];
        b = smem_ll[(ty*2) * 16 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 16 + (tx*2)];
        d = smem_ll[(ty*2+1) * 16 + (tx*2+1)];
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty, out2_x = tile_x * 8 + tx;
        if (out2_y < H4 && out2_x < W4) {
            write_subbands(level2, offset2, plane2, out2_y * W4 + out2_x, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 8 && ty < 8) smem_ll[ty * 8 + tx] = l2_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 3
    float l3_ll = 0, l3_lh, l3_hl, l3_hh;
    if (tx < 4 && ty < 4) {
        a = smem_ll[(ty*2) * 8 + (tx*2)];
        b = smem_ll[(ty*2) * 8 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 8 + (tx*2)];
        d = smem_ll[(ty*2+1) * 8 + (tx*2+1)];
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty, out3_x = tile_x * 4 + tx;
        if (out3_y < H8 && out3_x < W8) {
            write_subbands(level3, offset3, plane3, out3_y * W8 + out3_x, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 4 && ty < 4) smem_ll[ty * 4 + tx] = l3_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 4
    float l4_ll = 0, l4_lh, l4_hl, l4_hh;
    if (tx < 2 && ty < 2) {
        a = smem_ll[(ty*2) * 4 + (tx*2)];
        b = smem_ll[(ty*2) * 4 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 4 + (tx*2)];
        d = smem_ll[(ty*2+1) * 4 + (tx*2+1)];
        haar_step(a, b, c, d, l4_ll, l4_lh, l4_hl, l4_hh);
        
        int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
        int out4_y = tile_y * 2 + ty, out4_x = tile_x * 2 + tx;
        if (out4_y < H16 && out4_x < W16) {
            write_subbands(level4, offset4, plane4, out4_y * W16 + out4_x, l4_ll, l4_lh, l4_hl, l4_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 2 && ty < 2) smem_ll[ty * 2 + tx] = l4_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 5 (single thread)
    if (tid_linear == 0) {
        a = smem_ll[0];
        b = smem_ll[1];
        c = smem_ll[2];
        d = smem_ll[3];
        float l5_ll, l5_lh, l5_hl, l5_hh;
        haar_step(a, b, c, d, l5_ll, l5_lh, l5_hl, l5_hh);
        
        int plane5 = H32 * W32, offset5 = bc * 4 * plane5;
        int out5_idx = tile_y * W32 + tile_x;
        if (out5_idx < plane5) {
            write_subbands(level5, offset5, plane5, out5_idx, l5_ll, l5_lh, l5_hl, l5_hh);
        }
    }
}

// =============================================================================
// HALF PRECISION VARIANTS
// =============================================================================

// 2-Level Forward Cascade (Half)
kernel void haar2d_double_cascade_kernel_half(
    device const half* input [[buffer(0)]],
    device half* level1 [[buffer(1)]],
    device half* level2 [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    constant int& H2 [[buffer(5)]],
    constant int& W2 [[buffer(6)]],
    constant int& H4 [[buffer(7)]],
    constant int& W4 [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 4;
    int y_base = ty * 4;
    if (y_base >= H || x_base >= W) return;
    
    int in_offset = bc * H * W;
    
    // Read 4x4 input (convert half to float)
    float in_vals[4][4];
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            int y = min(y_base + dy, H - 1);
            int x = min(x_base + dx, W - 1);
            in_vals[dy][dx] = float(input[in_offset + y * W + x]);
        }
    }
    
    // Level 1: 4x4 -> 2x2
    float l1[4][2][2];
    for (int qy = 0; qy < 2; qy++) {
        for (int qx = 0; qx < 2; qx++) {
            float a = in_vals[qy*2][qx*2];
            float b = in_vals[qy*2][qx*2+1];
            float c = in_vals[qy*2+1][qx*2];
            float d = in_vals[qy*2+1][qx*2+1];
            haar_step(a, b, c, d, l1[0][qy][qx], l1[1][qy][qx], l1[2][qy][qx], l1[3][qy][qx]);
        }
    }
    
    // Write level 1 output
    int plane1 = H2 * W2;
    int offset1 = bc * 4 * plane1;
    int out1_y = ty * 2;
    int out1_x = tx * 2;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            if (out1_y + dy < H2 && out1_x + dx < W2) {
                int idx = (out1_y + dy) * W2 + (out1_x + dx);
                write_subbands_half(level1, offset1, plane1, idx,
                              l1[0][dy][dx], l1[1][dy][dx], l1[2][dy][dx], l1[3][dy][dx]);
            }
        }
    }
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    haar_step(l1[0][0][0], l1[0][0][1], l1[0][1][0], l1[0][1][1], l2_ll, l2_lh, l2_hl, l2_hh);
    
    int plane2 = H4 * W4;
    int offset2 = bc * 4 * plane2;
    int out2_y = ty;
    int out2_x = tx;
    
    if (out2_y < H4 && out2_x < W4) {
        int idx = out2_y * W4 + out2_x;
        write_subbands_half(level2, offset2, plane2, idx, l2_ll, l2_lh, l2_hl, l2_hh);
    }
}

// 3-Level Forward Cascade (Half)
kernel void haar2d_triple_cascade_kernel_half(
    device const half* input [[buffer(0)]],
    device half* level1 [[buffer(1)]],
    device half* level2 [[buffer(2)]],
    device half* level3 [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& H2 [[buffer(6)]],
    constant int& W2 [[buffer(7)]],
    constant int& H4 [[buffer(8)]],
    constant int& W4 [[buffer(9)]],
    constant int& H8 [[buffer(10)]],
    constant int& W8 [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
) {
    int tx = tid.x;
    int ty = tid.y;
    int bc = gid.z;
    
    int tile_x = gid.x / 16;
    int tile_y = gid.y / 16;
    int x_base = tile_x * 32 + tx * 2;
    int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    if (y_base < H && x_base < W) {
        int y0 = min(y_base, H-1);
        int y1 = min(y_base + 1, H-1);
        int x0 = min(x_base, W-1);
        int x1 = min(x_base + 1, W-1);
        
        a = float(input[in_offset + y0 * W + x0]);
        b = float(input[in_offset + y0 * W + x1]);
        c = float(input[in_offset + y1 * W + x0]);
        d = float(input[in_offset + y1 * W + x1]);
    } else {
        a = b = c = d = 0.0f;
    }
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2;
    int offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty;
    int out1_x = tile_x * 16 + tx;
    
    if (out1_y < H2 && out1_x < W2) {
        int idx = out1_y * W2 + out1_x;
        write_subbands_half(level1, offset1, plane1, idx, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    threadgroup float* smem_all_ll = smem;
    smem_all_ll[ty * 16 + tx] = l1_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_all_ll[(ty*2) * 16 + (tx*2)];
        b = smem_all_ll[(ty*2) * 16 + (tx*2+1)];
        c = smem_all_ll[(ty*2+1) * 16 + (tx*2)];
        d = smem_all_ll[(ty*2+1) * 16 + (tx*2+1)];
        
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4;
        int offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty;
        int out2_x = tile_x * 8 + tx;
        
        if (out2_y < H4 && out2_x < W4) {
            int idx = out2_y * W4 + out2_x;
            write_subbands_half(level2, offset2, plane2, idx, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float* smem_l2_ll = smem;
    if (tx < 8 && ty < 8) {
        smem_l2_ll[ty * 8 + tx] = l2_ll;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 3
    if (tx < 4 && ty < 4) {
        a = smem_l2_ll[(ty*2) * 8 + (tx*2)];
        b = smem_l2_ll[(ty*2) * 8 + (tx*2+1)];
        c = smem_l2_ll[(ty*2+1) * 8 + (tx*2)];
        d = smem_l2_ll[(ty*2+1) * 8 + (tx*2+1)];
        
        float l3_ll, l3_lh, l3_hl, l3_hh;
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8;
        int offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty;
        int out3_x = tile_x * 4 + tx;
        
        if (out3_y < H8 && out3_x < W8) {
            int idx = out3_y * W8 + out3_x;
            write_subbands_half(level3, offset3, plane3, idx, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
}

// 4-Level Forward Cascade (Half)
kernel void haar2d_quad_cascade_kernel_half(
    device const half* input [[buffer(0)]],
    device half* level1 [[buffer(1)]],
    device half* level2 [[buffer(2)]],
    device half* level3 [[buffer(3)]],
    device half* level4 [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& W [[buffer(6)]],
    constant int& H2 [[buffer(7)]],
    constant int& W2 [[buffer(8)]],
    constant int& H4 [[buffer(9)]],
    constant int& W4 [[buffer(10)]],
    constant int& H8 [[buffer(11)]],
    constant int& W8 [[buffer(12)]],
    constant int& H16 [[buffer(13)]],
    constant int& W16 [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
) {
    int tx = tid.x;
    int ty = tid.y;
    int bc = gid.z;
    int tile_x = gid.x / 16;
    int tile_y = gid.y / 16;
    int x_base = tile_x * 32 + tx * 2;
    int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    int y0 = min(y_base, H-1);
    int y1 = min(y_base+1, H-1);
    int x0 = min(x_base, W-1);
    int x1 = min(x_base+1, W-1);
    
    a = float(input[in_offset + y0 * W + x0]);
    b = float(input[in_offset + y0 * W + x1]);
    c = float(input[in_offset + y1 * W + x0]);
    d = float(input[in_offset + y1 * W + x1]);
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty, out1_x = tile_x * 16 + tx;
    if (out1_y < H2 && out1_x < W2) {
        write_subbands_half(level1, offset1, plane1, out1_y * W2 + out1_x, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    threadgroup float* smem_ll = smem;
    smem_ll[ty * 16 + tx] = l1_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_ll[(ty*2) * 16 + (tx*2)];
        b = smem_ll[(ty*2) * 16 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 16 + (tx*2)];
        d = smem_ll[(ty*2+1) * 16 + (tx*2+1)];
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty, out2_x = tile_x * 8 + tx;
        if (out2_y < H4 && out2_x < W4) {
            write_subbands_half(level2, offset2, plane2, out2_y * W4 + out2_x, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 8 && ty < 8) smem_ll[ty * 8 + tx] = l2_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 3
    float l3_ll = 0, l3_lh, l3_hl, l3_hh;
    if (tx < 4 && ty < 4) {
        a = smem_ll[(ty*2) * 8 + (tx*2)];
        b = smem_ll[(ty*2) * 8 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 8 + (tx*2)];
        d = smem_ll[(ty*2+1) * 8 + (tx*2+1)];
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty, out3_x = tile_x * 4 + tx;
        if (out3_y < H8 && out3_x < W8) {
            write_subbands_half(level3, offset3, plane3, out3_y * W8 + out3_x, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 4 && ty < 4) smem_ll[ty * 4 + tx] = l3_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 4
    if (tx < 2 && ty < 2) {
        a = smem_ll[(ty*2) * 4 + (tx*2)];
        b = smem_ll[(ty*2) * 4 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 4 + (tx*2)];
        d = smem_ll[(ty*2+1) * 4 + (tx*2+1)];
        float l4_ll, l4_lh, l4_hl, l4_hh;
        haar_step(a, b, c, d, l4_ll, l4_lh, l4_hl, l4_hh);
        
        int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
        int out4_y = tile_y * 2 + ty, out4_x = tile_x * 2 + tx;
        if (out4_y < H16 && out4_x < W16) {
            write_subbands_half(level4, offset4, plane4, out4_y * W16 + out4_x, l4_ll, l4_lh, l4_hl, l4_hh);
        }
    }
}

// 5-Level Forward Cascade (Half)
kernel void haar2d_quint_cascade_kernel_half(
    device const half* input [[buffer(0)]],
    device half* level1 [[buffer(1)]],
    device half* level2 [[buffer(2)]],
    device half* level3 [[buffer(3)]],
    device half* level4 [[buffer(4)]],
    device half* level5 [[buffer(5)]],
    constant int& H [[buffer(6)]],
    constant int& W [[buffer(7)]],
    constant int& H2 [[buffer(8)]],
    constant int& W2 [[buffer(9)]],
    constant int& H4 [[buffer(10)]],
    constant int& W4 [[buffer(11)]],
    constant int& H8 [[buffer(12)]],
    constant int& W8 [[buffer(13)]],
    constant int& H16 [[buffer(14)]],
    constant int& W16 [[buffer(15)]],
    constant int& H32 [[buffer(16)]],
    constant int& W32 [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
) {
    int tx = tid.x;
    int ty = tid.y;
    int tid_linear = ty * 16 + tx;
    int bc = gid.z;
    int tile_x = gid.x / 16;
    int tile_y = gid.y / 16;
    int x_base = tile_x * 32 + tx * 2;
    int y_base = tile_y * 32 + ty * 2;
    
    if (tile_y * 32 >= H || tile_x * 32 >= W) return;
    
    int in_offset = bc * H * W;
    
    // Level 1
    float a, b, c, d;
    int y0 = min(y_base, H-1);
    int y1 = min(y_base+1, H-1);
    int x0 = min(x_base, W-1);
    int x1 = min(x_base+1, W-1);
    
    a = float(input[in_offset + y0 * W + x0]);
    b = float(input[in_offset + y0 * W + x1]);
    c = float(input[in_offset + y1 * W + x0]);
    d = float(input[in_offset + y1 * W + x1]);
    
    float l1_ll, l1_lh, l1_hl, l1_hh;
    haar_step(a, b, c, d, l1_ll, l1_lh, l1_hl, l1_hh);
    
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int out1_y = tile_y * 16 + ty, out1_x = tile_x * 16 + tx;
    if (out1_y < H2 && out1_x < W2) {
        write_subbands_half(level1, offset1, plane1, out1_y * W2 + out1_x, l1_ll, l1_lh, l1_hl, l1_hh);
    }
    
    threadgroup float* smem_ll = smem;
    smem_ll[ty * 16 + tx] = l1_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 2
    float l2_ll = 0, l2_lh, l2_hl, l2_hh;
    if (tx < 8 && ty < 8) {
        a = smem_ll[(ty*2) * 16 + (tx*2)];
        b = smem_ll[(ty*2) * 16 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 16 + (tx*2)];
        d = smem_ll[(ty*2+1) * 16 + (tx*2+1)];
        haar_step(a, b, c, d, l2_ll, l2_lh, l2_hl, l2_hh);
        
        int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
        int out2_y = tile_y * 8 + ty, out2_x = tile_x * 8 + tx;
        if (out2_y < H4 && out2_x < W4) {
            write_subbands_half(level2, offset2, plane2, out2_y * W4 + out2_x, l2_ll, l2_lh, l2_hl, l2_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 8 && ty < 8) smem_ll[ty * 8 + tx] = l2_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 3
    float l3_ll = 0, l3_lh, l3_hl, l3_hh;
    if (tx < 4 && ty < 4) {
        a = smem_ll[(ty*2) * 8 + (tx*2)];
        b = smem_ll[(ty*2) * 8 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 8 + (tx*2)];
        d = smem_ll[(ty*2+1) * 8 + (tx*2+1)];
        haar_step(a, b, c, d, l3_ll, l3_lh, l3_hl, l3_hh);
        
        int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
        int out3_y = tile_y * 4 + ty, out3_x = tile_x * 4 + tx;
        if (out3_y < H8 && out3_x < W8) {
            write_subbands_half(level3, offset3, plane3, out3_y * W8 + out3_x, l3_ll, l3_lh, l3_hl, l3_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 4 && ty < 4) smem_ll[ty * 4 + tx] = l3_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 4
    float l4_ll = 0, l4_lh, l4_hl, l4_hh;
    if (tx < 2 && ty < 2) {
        a = smem_ll[(ty*2) * 4 + (tx*2)];
        b = smem_ll[(ty*2) * 4 + (tx*2+1)];
        c = smem_ll[(ty*2+1) * 4 + (tx*2)];
        d = smem_ll[(ty*2+1) * 4 + (tx*2+1)];
        haar_step(a, b, c, d, l4_ll, l4_lh, l4_hl, l4_hh);
        
        int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
        int out4_y = tile_y * 2 + ty, out4_x = tile_x * 2 + tx;
        if (out4_y < H16 && out4_x < W16) {
            write_subbands_half(level4, offset4, plane4, out4_y * W16 + out4_x, l4_ll, l4_lh, l4_hl, l4_hh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tx < 2 && ty < 2) smem_ll[ty * 2 + tx] = l4_ll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Level 5 (single thread)
    if (tid_linear == 0) {
        a = smem_ll[0];
        b = smem_ll[1];
        c = smem_ll[2];
        d = smem_ll[3];
        float l5_ll, l5_lh, l5_hl, l5_hh;
        haar_step(a, b, c, d, l5_ll, l5_lh, l5_hl, l5_hh);
        
        int plane5 = H32 * W32, offset5 = bc * 4 * plane5;
        int out5_idx = tile_y * W32 + tile_x;
        if (out5_idx < plane5) {
            write_subbands_half(level5, offset5, plane5, out5_idx, l5_ll, l5_lh, l5_hl, l5_hh);
        }
    }
}

