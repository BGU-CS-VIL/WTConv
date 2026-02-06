#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Haar 2D Wavelet Transform - Inverse & Inverse Backward (Metal)
// Inverse: (B, C, 4, H/2, W/2) -> (B, C, H, W)
// Supports float and half data types
// =============================================================================

// -----------------------------------------------------------------------------
// Inverse Haar Transform: (B, C, 4, H/2, W/2) -> (B, C, H, W)
// Reconstructs original image from wavelet coefficients
// -----------------------------------------------------------------------------
kernel void haar2d_inverse_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& W [[buffer(3)]],
    constant int& H2 [[buffer(4)]],
    constant int& W2 [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;  // Combined batch*channel index
    
    if (y >= H2 || x >= W2) return;
    
    int plane = H2 * W2;
    int in_offset = bc * 4 * plane;
    int idx = y * W2 + x;
    
    float ll = input[in_offset + 0 * plane + idx];
    float lh = input[in_offset + 1 * plane + idx];
    float hl = input[in_offset + 2 * plane + idx];
    float hh = input[in_offset + 3 * plane + idx];
    
    // Inverse Haar: reconstruct 2x2 block
    float a = 0.5f * (ll + lh + hl + hh);
    float b = 0.5f * (ll + lh - hl - hh);
    float c = 0.5f * (ll - lh + hl - hh);
    float d = 0.5f * (ll - lh - hl + hh);
    
    int x0 = 2 * x;
    int x1 = 2 * x + 1;
    int y0 = 2 * y;
    int y1 = 2 * y + 1;
    int out_offset = bc * H * W;
    
    output[out_offset + y0 * W + x0] = a;
    if (x1 < W) output[out_offset + y0 * W + x1] = b;
    if (y1 < H) output[out_offset + y1 * W + x0] = c;
    if (x1 < W && y1 < H) output[out_offset + y1 * W + x1] = d;
}

// Half precision version
kernel void haar2d_inverse_kernel_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
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
    int in_offset = bc * 4 * plane;
    int idx = y * W2 + x;
    
    float ll = float(input[in_offset + 0 * plane + idx]);
    float lh = float(input[in_offset + 1 * plane + idx]);
    float hl = float(input[in_offset + 2 * plane + idx]);
    float hh = float(input[in_offset + 3 * plane + idx]);
    
    float a = 0.5f * (ll + lh + hl + hh);
    float b = 0.5f * (ll + lh - hl - hh);
    float c = 0.5f * (ll - lh + hl - hh);
    float d = 0.5f * (ll - lh - hl + hh);
    
    int x0 = 2 * x;
    int x1 = 2 * x + 1;
    int y0 = 2 * y;
    int y1 = 2 * y + 1;
    int out_offset = bc * H * W;
    
    output[out_offset + y0 * W + x0] = half(a);
    if (x1 < W) output[out_offset + y0 * W + x1] = half(b);
    if (y1 < H) output[out_offset + y1 * W + x0] = half(c);
    if (x1 < W && y1 < H) output[out_offset + y1 * W + x1] = half(d);
}

// -----------------------------------------------------------------------------
// Backward for Inverse Haar: (B, C, H, W) -> (B, C, 4, H/2, W/2)
// This is essentially the forward Haar transform
// -----------------------------------------------------------------------------
kernel void haar2d_inverse_backward_kernel(
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
    
    int x0 = 2 * x;
    int x1 = min(2 * x + 1, W - 1);
    int y0 = 2 * y;
    int y1 = min(2 * y + 1, H - 1);
    
    int grad_offset = bc * H * W;
    float a = grad_output[grad_offset + y0 * W + x0];
    float b = grad_output[grad_offset + y0 * W + x1];
    float c = grad_output[grad_offset + y1 * W + x0];
    float d = grad_output[grad_offset + y1 * W + x1];
    
    float sum_ac = a + c;
    float sum_bd = b + d;
    float diff_ac = a - c;
    float diff_bd = b - d;
    
    int out_idx = y * W2 + x;
    int plane = H2 * W2;
    int out_offset = bc * 4 * plane;
    
    grad_input[out_offset + 0 * plane + out_idx] = 0.5f * (sum_ac + sum_bd);   // LL
    grad_input[out_offset + 1 * plane + out_idx] = 0.5f * (diff_ac + diff_bd); // LH
    grad_input[out_offset + 2 * plane + out_idx] = 0.5f * (sum_ac - sum_bd);   // HL
    grad_input[out_offset + 3 * plane + out_idx] = 0.5f * (diff_ac - diff_bd); // HH
}

// Half precision version
kernel void haar2d_inverse_backward_kernel_half(
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
    
    int x0 = 2 * x;
    int x1 = min(2 * x + 1, W - 1);
    int y0 = 2 * y;
    int y1 = min(2 * y + 1, H - 1);
    
    int grad_offset = bc * H * W;
    float a = float(grad_output[grad_offset + y0 * W + x0]);
    float b = float(grad_output[grad_offset + y0 * W + x1]);
    float c = float(grad_output[grad_offset + y1 * W + x0]);
    float d = float(grad_output[grad_offset + y1 * W + x1]);
    
    float sum_ac = a + c;
    float sum_bd = b + d;
    float diff_ac = a - c;
    float diff_bd = b - d;
    
    int out_idx = y * W2 + x;
    int plane = H2 * W2;
    int out_offset = bc * 4 * plane;
    
    grad_input[out_offset + 0 * plane + out_idx] = half(0.5f * (sum_ac + sum_bd));
    grad_input[out_offset + 1 * plane + out_idx] = half(0.5f * (diff_ac + diff_bd));
    grad_input[out_offset + 2 * plane + out_idx] = half(0.5f * (sum_ac - sum_bd));
    grad_input[out_offset + 3 * plane + out_idx] = half(0.5f * (diff_ac - diff_bd));
}
