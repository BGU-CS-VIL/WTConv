#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Haar 2D Wavelet Transform - Single Level Forward & Backward (Metal)
// Supports multi-channel input: (B, C, H, W) -> (B, C, 4, H/2, W/2)
// Supports float and half data types
// =============================================================================

// -----------------------------------------------------------------------------
// Forward: (B, C, H, W) -> (B, C, 4, H/2, W/2)
// Applies Haar transform independently to each channel
// -----------------------------------------------------------------------------
kernel void haar2d_forward_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& C [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    constant int& H2 [[buffer(5)]],
    constant int& W2 [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;  // Combined batch*channel index
    
    if (y >= H2 || x >= W2) return;
    
    int x0 = 2 * x;
    int x1 = min(2 * x + 1, W - 1);
    int y0 = 2 * y;
    int y1 = min(2 * y + 1, H - 1);
    
    int in_offset = bc * H * W;
    float a = input[in_offset + y0 * W + x0];
    float b = input[in_offset + y0 * W + x1];
    float c = input[in_offset + y1 * W + x0];
    float d = input[in_offset + y1 * W + x1];
    
    float sum_ac = a + c;
    float sum_bd = b + d;
    float diff_ac = a - c;
    float diff_bd = b - d;
    
    int out_idx = y * W2 + x;
    int plane = H2 * W2;
    int out_offset = bc * 4 * plane;
    
    output[out_offset + 0 * plane + out_idx] = 0.5f * (sum_ac + sum_bd);   // LL
    output[out_offset + 1 * plane + out_idx] = 0.5f * (diff_ac + diff_bd); // LH
    output[out_offset + 2 * plane + out_idx] = 0.5f * (sum_ac - sum_bd);   // HL
    output[out_offset + 3 * plane + out_idx] = 0.5f * (diff_ac - diff_bd); // HH
}

// Half precision version
kernel void haar2d_forward_kernel_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant int& C [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    constant int& H2 [[buffer(5)]],
    constant int& W2 [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    
    if (y >= H2 || x >= W2) return;
    
    int x0 = 2 * x;
    int x1 = min(2 * x + 1, W - 1);
    int y0 = 2 * y;
    int y1 = min(2 * y + 1, H - 1);
    
    int in_offset = bc * H * W;
    float a = float(input[in_offset + y0 * W + x0]);
    float b = float(input[in_offset + y0 * W + x1]);
    float c = float(input[in_offset + y1 * W + x0]);
    float d = float(input[in_offset + y1 * W + x1]);
    
    float sum_ac = a + c;
    float sum_bd = b + d;
    float diff_ac = a - c;
    float diff_bd = b - d;
    
    int out_idx = y * W2 + x;
    int plane = H2 * W2;
    int out_offset = bc * 4 * plane;
    
    output[out_offset + 0 * plane + out_idx] = half(0.5f * (sum_ac + sum_bd));
    output[out_offset + 1 * plane + out_idx] = half(0.5f * (diff_ac + diff_bd));
    output[out_offset + 2 * plane + out_idx] = half(0.5f * (sum_ac - sum_bd));
    output[out_offset + 3 * plane + out_idx] = half(0.5f * (diff_ac - diff_bd));
}

// -----------------------------------------------------------------------------
// Backward: (B, C, 4, H/2, W/2) -> (B, C, H, W)
// -----------------------------------------------------------------------------
kernel void haar2d_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device float* grad_input [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& W [[buffer(3)]],
    constant int& H2 [[buffer(4)]],
    constant int& W2 [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    
    if (y >= H2 || x >= W2) return;
    
    int plane = H2 * W2;
    int grad_offset = bc * 4 * plane;
    int idx = y * W2 + x;
    
    float g_ll = grad_output[grad_offset + 0 * plane + idx];
    float g_lh = grad_output[grad_offset + 1 * plane + idx];
    float g_hl = grad_output[grad_offset + 2 * plane + idx];
    float g_hh = grad_output[grad_offset + 3 * plane + idx];
    
    float grad_a = 0.5f * (g_ll + g_lh + g_hl + g_hh);
    float grad_b = 0.5f * (g_ll + g_lh - g_hl - g_hh);
    float grad_c = 0.5f * (g_ll - g_lh + g_hl - g_hh);
    float grad_d = 0.5f * (g_ll - g_lh - g_hl + g_hh);
    
    int x0 = 2 * x;
    int x1 = 2 * x + 1;
    int y0 = 2 * y;
    int y1 = 2 * y + 1;
    int in_offset = bc * H * W;
    
    grad_input[in_offset + y0 * W + x0] = grad_a;
    if (x1 < W) grad_input[in_offset + y0 * W + x1] = grad_b;
    if (y1 < H) grad_input[in_offset + y1 * W + x0] = grad_c;
    if (x1 < W && y1 < H) grad_input[in_offset + y1 * W + x1] = grad_d;
}

// Half precision backward
kernel void haar2d_backward_kernel_half(
    device const half* grad_output [[buffer(0)]],
    device half* grad_input [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& W [[buffer(3)]],
    constant int& H2 [[buffer(4)]],
    constant int& W2 [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    
    if (y >= H2 || x >= W2) return;
    
    int plane = H2 * W2;
    int grad_offset = bc * 4 * plane;
    int idx = y * W2 + x;
    
    float g_ll = float(grad_output[grad_offset + 0 * plane + idx]);
    float g_lh = float(grad_output[grad_offset + 1 * plane + idx]);
    float g_hl = float(grad_output[grad_offset + 2 * plane + idx]);
    float g_hh = float(grad_output[grad_offset + 3 * plane + idx]);
    
    float grad_a = 0.5f * (g_ll + g_lh + g_hl + g_hh);
    float grad_b = 0.5f * (g_ll + g_lh - g_hl - g_hh);
    float grad_c = 0.5f * (g_ll - g_lh + g_hl - g_hh);
    float grad_d = 0.5f * (g_ll - g_lh - g_hl + g_hh);
    
    int x0 = 2 * x;
    int x1 = 2 * x + 1;
    int y0 = 2 * y;
    int y1 = 2 * y + 1;
    int in_offset = bc * H * W;
    
    grad_input[in_offset + y0 * W + x0] = half(grad_a);
    if (x1 < W) grad_input[in_offset + y0 * W + x1] = half(grad_b);
    if (y1 < H) grad_input[in_offset + y1 * W + x0] = half(grad_c);
    if (x1 < W && y1 < H) grad_input[in_offset + y1 * W + x1] = half(grad_d);
}
