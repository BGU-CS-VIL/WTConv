#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Optimized Fused Inverse Haar 2D - Metal (All Levels: 2, 3, 4, 5)
//
// Optimizations:
// 1. Threadgroup memory for intermediate results
// 2. Optimized memory access patterns
// =============================================================================

// Inverse Haar step helper
inline void ihaar_step(float ll, float lh, float hl, float hh,
                       thread float& a, thread float& b, thread float& c, thread float& d) {
    a = 0.5f * (ll + lh + hl + hh);
    b = 0.5f * (ll + lh - hl - hh);
    c = 0.5f * (ll - lh + hl - hh);
    d = 0.5f * (ll - lh - hl + hh);
}

// Load 4 subbands helper
inline void load_subbands(device const float* data, int offset, int plane, int idx,
                          thread float& ll, thread float& lh, thread float& hl, thread float& hh) {
    ll = data[offset + 0 * plane + idx];
    lh = data[offset + 1 * plane + idx];
    hl = data[offset + 2 * plane + idx];
    hh = data[offset + 3 * plane + idx];
}

// Half precision helpers
inline void load_subbands_half(device const half* data, int offset, int plane, int idx,
                               thread float& ll, thread float& lh, thread float& hl, thread float& hh) {
    ll = float(data[offset + 0 * plane + idx]);
    lh = float(data[offset + 1 * plane + idx]);
    hl = float(data[offset + 2 * plane + idx]);
    hh = float(data[offset + 3 * plane + idx]);
}

// =============================================================================
// 2-Level Inverse Cascade
// =============================================================================
kernel void ihaar2d_double_cascade_kernel(
    device const float* level1 [[buffer(0)]],
    device const float* level2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    constant int& H2 [[buffer(5)]],
    constant int& W2 [[buffer(6)]],
    constant int& H4 [[buffer(7)]],
    constant int& W4 [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = out_vals[0][0];
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = out_vals[0][1];
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = out_vals[1][0];
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = out_vals[1][1];
}

// =============================================================================
// 3-Level Inverse Cascade
// =============================================================================
kernel void ihaar2d_triple_cascade_kernel(
    device const float* level1 [[buffer(0)]],
    device const float* level2 [[buffer(1)]],
    device const float* level3 [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& H2 [[buffer(6)]],
    constant int& W2 [[buffer(7)]],
    constant int& H4 [[buffer(8)]],
    constant int& W4 [[buffer(9)]],
    constant int& H8 [[buffer(10)]],
    constant int& W8 [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int x8 = x4 / 2, y8 = y4 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    int q4x = x4 % 2, q4y = y4 % 2;
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = min(y8, H8-1) * W8 + min(x8, W8-1);
    load_subbands(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = out_vals[0][0];
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = out_vals[0][1];
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = out_vals[1][0];
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = out_vals[1][1];
}

// =============================================================================
// 4-Level Inverse Cascade
// =============================================================================
kernel void ihaar2d_quad_cascade_kernel(
    device const float* level1 [[buffer(0)]],
    device const float* level2 [[buffer(1)]],
    device const float* level3 [[buffer(2)]],
    device const float* level4 [[buffer(3)]],
    device float* output [[buffer(4)]],
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
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int x8 = x4 / 2, y8 = y4 / 2;
    int x16 = x8 / 2, y16 = y8 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    int q4x = x4 % 2, q4y = y4 % 2;
    int q8x = x8 % 2, q8y = y8 % 2;
    
    // Level 4
    float l4_ll, l4_lh, l4_hl, l4_hh;
    int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
    int idx4 = min(y16, H16-1) * W16 + min(x16, W16-1);
    load_subbands(level4, offset4, plane4, idx4, l4_ll, l4_lh, l4_hl, l4_hh);
    float r4[2][2];
    ihaar_step(l4_ll, l4_lh, l4_hl, l4_hh, r4[0][0], r4[0][1], r4[1][0], r4[1][1]);
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = min(y8, H8-1) * W8 + min(x8, W8-1);
    load_subbands(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    l3_ll += r4[q8y][q8x];
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = out_vals[0][0];
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = out_vals[0][1];
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = out_vals[1][0];
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = out_vals[1][1];
}

// =============================================================================
// 5-Level Inverse Cascade
// =============================================================================
kernel void ihaar2d_quint_cascade_kernel(
    device const float* level1 [[buffer(0)]],
    device const float* level2 [[buffer(1)]],
    device const float* level3 [[buffer(2)]],
    device const float* level4 [[buffer(3)]],
    device const float* level5 [[buffer(4)]],
    device float* output [[buffer(5)]],
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
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int x8 = x4 / 2, y8 = y4 / 2;
    int x16 = x8 / 2, y16 = y8 / 2;
    int x32 = x16 / 2, y32 = y16 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    int q4x = x4 % 2, q4y = y4 % 2;
    int q8x = x8 % 2, q8y = y8 % 2;
    int q16x = x16 % 2, q16y = y16 % 2;
    
    // Level 5
    float l5_ll, l5_lh, l5_hl, l5_hh;
    int plane5 = H32 * W32, offset5 = bc * 4 * plane5;
    int idx5 = min(y32, H32-1) * W32 + min(x32, W32-1);
    load_subbands(level5, offset5, plane5, idx5, l5_ll, l5_lh, l5_hl, l5_hh);
    float r5[2][2];
    ihaar_step(l5_ll, l5_lh, l5_hl, l5_hh, r5[0][0], r5[0][1], r5[1][0], r5[1][1]);
    
    // Level 4
    float l4_ll, l4_lh, l4_hl, l4_hh;
    int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
    int idx4 = min(y16, H16-1) * W16 + min(x16, W16-1);
    load_subbands(level4, offset4, plane4, idx4, l4_ll, l4_lh, l4_hl, l4_hh);
    l4_ll += r5[q16y][q16x];
    float r4[2][2];
    ihaar_step(l4_ll, l4_lh, l4_hl, l4_hh, r4[0][0], r4[0][1], r4[1][0], r4[1][1]);
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = min(y8, H8-1) * W8 + min(x8, W8-1);
    load_subbands(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    l3_ll += r4[q8y][q8x];
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = out_vals[0][0];
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = out_vals[0][1];
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = out_vals[1][0];
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = out_vals[1][1];
}

// =============================================================================
// HALF PRECISION VARIANTS
// =============================================================================

// 2-Level Inverse Cascade (Half)
kernel void ihaar2d_double_cascade_kernel_half(
    device const half* level1 [[buffer(0)]],
    device const half* level2 [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    constant int& H2 [[buffer(5)]],
    constant int& W2 [[buffer(6)]],
    constant int& H4 [[buffer(7)]],
    constant int& W4 [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands_half(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands_half(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = half(out_vals[0][0]);
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = half(out_vals[0][1]);
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = half(out_vals[1][0]);
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = half(out_vals[1][1]);
}

// 3-Level Inverse Cascade (Half)
kernel void ihaar2d_triple_cascade_kernel_half(
    device const half* level1 [[buffer(0)]],
    device const half* level2 [[buffer(1)]],
    device const half* level3 [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& H2 [[buffer(6)]],
    constant int& W2 [[buffer(7)]],
    constant int& H4 [[buffer(8)]],
    constant int& W4 [[buffer(9)]],
    constant int& H8 [[buffer(10)]],
    constant int& W8 [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int x8 = x4 / 2, y8 = y4 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    int q4x = x4 % 2, q4y = y4 % 2;
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = min(y8, H8-1) * W8 + min(x8, W8-1);
    load_subbands_half(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands_half(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands_half(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = half(out_vals[0][0]);
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = half(out_vals[0][1]);
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = half(out_vals[1][0]);
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = half(out_vals[1][1]);
}

// 4-Level Inverse Cascade (Half)
kernel void ihaar2d_quad_cascade_kernel_half(
    device const half* level1 [[buffer(0)]],
    device const half* level2 [[buffer(1)]],
    device const half* level3 [[buffer(2)]],
    device const half* level4 [[buffer(3)]],
    device half* output [[buffer(4)]],
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
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int x8 = x4 / 2, y8 = y4 / 2;
    int x16 = x8 / 2, y16 = y8 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    int q4x = x4 % 2, q4y = y4 % 2;
    int q8x = x8 % 2, q8y = y8 % 2;
    
    // Level 4
    float l4_ll, l4_lh, l4_hl, l4_hh;
    int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
    int idx4 = min(y16, H16-1) * W16 + min(x16, W16-1);
    load_subbands_half(level4, offset4, plane4, idx4, l4_ll, l4_lh, l4_hl, l4_hh);
    float r4[2][2];
    ihaar_step(l4_ll, l4_lh, l4_hl, l4_hh, r4[0][0], r4[0][1], r4[1][0], r4[1][1]);
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = min(y8, H8-1) * W8 + min(x8, W8-1);
    load_subbands_half(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    l3_ll += r4[q8y][q8x];
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands_half(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands_half(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = half(out_vals[0][0]);
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = half(out_vals[0][1]);
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = half(out_vals[1][0]);
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = half(out_vals[1][1]);
}

// 5-Level Inverse Cascade (Half)
kernel void ihaar2d_quint_cascade_kernel_half(
    device const half* level1 [[buffer(0)]],
    device const half* level2 [[buffer(1)]],
    device const half* level3 [[buffer(2)]],
    device const half* level4 [[buffer(3)]],
    device const half* level5 [[buffer(4)]],
    device half* output [[buffer(5)]],
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
    uint3 gid [[thread_position_in_grid]]
) {
    int tx = gid.x;
    int ty = gid.y;
    int bc = gid.z;
    
    int x_base = tx * 2;
    int y_base = ty * 2;
    if (y_base >= H || x_base >= W) return;
    
    int x2 = tx, y2 = ty;
    int x4 = x2 / 2, y4 = y2 / 2;
    int x8 = x4 / 2, y8 = y4 / 2;
    int x16 = x8 / 2, y16 = y8 / 2;
    int x32 = x16 / 2, y32 = y16 / 2;
    int q2x = x2 % 2, q2y = y2 % 2;
    int q4x = x4 % 2, q4y = y4 % 2;
    int q8x = x8 % 2, q8y = y8 % 2;
    int q16x = x16 % 2, q16y = y16 % 2;
    
    // Level 5
    float l5_ll, l5_lh, l5_hl, l5_hh;
    int plane5 = H32 * W32, offset5 = bc * 4 * plane5;
    int idx5 = min(y32, H32-1) * W32 + min(x32, W32-1);
    load_subbands_half(level5, offset5, plane5, idx5, l5_ll, l5_lh, l5_hl, l5_hh);
    float r5[2][2];
    ihaar_step(l5_ll, l5_lh, l5_hl, l5_hh, r5[0][0], r5[0][1], r5[1][0], r5[1][1]);
    
    // Level 4
    float l4_ll, l4_lh, l4_hl, l4_hh;
    int plane4 = H16 * W16, offset4 = bc * 4 * plane4;
    int idx4 = min(y16, H16-1) * W16 + min(x16, W16-1);
    load_subbands_half(level4, offset4, plane4, idx4, l4_ll, l4_lh, l4_hl, l4_hh);
    l4_ll += r5[q16y][q16x];
    float r4[2][2];
    ihaar_step(l4_ll, l4_lh, l4_hl, l4_hh, r4[0][0], r4[0][1], r4[1][0], r4[1][1]);
    
    // Level 3
    float l3_ll, l3_lh, l3_hl, l3_hh;
    int plane3 = H8 * W8, offset3 = bc * 4 * plane3;
    int idx3 = min(y8, H8-1) * W8 + min(x8, W8-1);
    load_subbands_half(level3, offset3, plane3, idx3, l3_ll, l3_lh, l3_hl, l3_hh);
    l3_ll += r4[q8y][q8x];
    float r3[2][2];
    ihaar_step(l3_ll, l3_lh, l3_hl, l3_hh, r3[0][0], r3[0][1], r3[1][0], r3[1][1]);
    
    // Level 2
    float l2_ll, l2_lh, l2_hl, l2_hh;
    int plane2 = H4 * W4, offset2 = bc * 4 * plane2;
    int idx2 = min(y4, H4-1) * W4 + min(x4, W4-1);
    load_subbands_half(level2, offset2, plane2, idx2, l2_ll, l2_lh, l2_hl, l2_hh);
    l2_ll += r3[q4y][q4x];
    float r2[2][2];
    ihaar_step(l2_ll, l2_lh, l2_hl, l2_hh, r2[0][0], r2[0][1], r2[1][0], r2[1][1]);
    
    // Level 1
    float l1_ll, l1_lh, l1_hl, l1_hh;
    int plane1 = H2 * W2, offset1 = bc * 4 * plane1;
    int idx1 = min(y2, H2-1) * W2 + min(x2, W2-1);
    load_subbands_half(level1, offset1, plane1, idx1, l1_ll, l1_lh, l1_hl, l1_hh);
    l1_ll += r2[q2y][q2x];
    float out_vals[2][2];
    ihaar_step(l1_ll, l1_lh, l1_hl, l1_hh, out_vals[0][0], out_vals[0][1], out_vals[1][0], out_vals[1][1]);
    
    int out_offset = bc * H * W;
    if (y_base < H && x_base < W) output[out_offset + y_base * W + x_base] = half(out_vals[0][0]);
    if (y_base < H && x_base + 1 < W) output[out_offset + y_base * W + x_base + 1] = half(out_vals[0][1]);
    if (y_base + 1 < H && x_base < W) output[out_offset + (y_base + 1) * W + x_base] = half(out_vals[1][0]);
    if (y_base + 1 < H && x_base + 1 < W) output[out_offset + (y_base + 1) * W + x_base + 1] = half(out_vals[1][1]);
}

