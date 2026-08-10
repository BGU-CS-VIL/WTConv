// =============================================================================
// Fused Haar -> depthwise conv -> scale (weight fusion with Haar)
//
// The wavelet branch of WTConv computes, per level,
//     coeffs = haar(x)                        (B, C, 4, H/2, W/2)
//     out    = scale * conv_depthwise(coeffs)
// Materialising `coeffs` costs a full round trip through HBM. Instead we note
// that each kernel tap (kh, kw) of the depthwise conv reads exactly one Haar
// coefficient, which comes from exactly one 2x2 input block. So for output
// position (h2, w2) and tap (kh, kw) we load that 2x2 block once, form the four
// Haar partial sums, and FMA each into its own subband accumulator with a
// single weight:
//     out[s] = sum_{kh,kw} w_fused[c, s, kh, kw] * P_s(2*(h2+kh-R), 2*(w2+kw-R))
// where R = K/2, P_LL/P_LH/P_HL/P_HH are the partial sums of that 2x2 block and
// w_fused = scale * weight is folded on the host (see compute_scaled_weight).
//
// Weight memory is (C, 4, K, K) rather than the (C, 4, 2K, 2K) "effective
// kernel" a naive fusion would build: 4x less weight traffic, 4x fewer FMAs.
//
// Each block stages the partial sums for its output tile (plus a K-1 halo) in
// shared memory, so the butterfly runs once per 2x2 block and each input pixel
// is fetched once per tile that needs it, instead of the K*K times a
// thread-per-output formulation would read it. Halo positions are re-staged by
// each neighbouring block, but they are small and spatially local, so those
// re-fetches hit L2 rather than HBM.
//
// Grid: one block per (b, c, output tile); block = (TILE_W, TILE_H) threads,
// one output position (4 subbands) per thread.
// =============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "haar_common.cuh"

#define TILE_W 32
#define TILE_H 8
#define TILE_THREADS (TILE_W * TILE_H)

// -----------------------------------------------------------------------------
// Forward: (B, C, H, W) -> (B, C, 4, H2, W2) [+ raw LL for the next level]
// -----------------------------------------------------------------------------
template<typename T, int K>
__global__ void fused_haar_conv_scale_kernel(
    const T* __restrict__ input,         // (B, C, H, W), contiguous
    const float* __restrict__ fused_w,   // (C, 4, K, K), contiguous
    T* __restrict__ output,              // (B, C, 4, H2, W2), contiguous
    T* __restrict__ ll_out,              // (B, C, H2, W2) or nullptr
    int C, int H, int W, int H2, int W2,
    int tiles_x, int tiles_y
) {
    constexpr int R = K / 2;
    constexpr int SH = TILE_H + K - 1;   // haloed tile height (coefficient grid)
    constexpr int SW = TILE_W + K - 1;
    constexpr int SPLANE = SH * SW;
    constexpr int WCOUNT = 4 * K * K;

    __shared__ float sh_p[4][SPLANE];    // Haar partial sums, one plane per subband
    __shared__ float sh_w[WCOUNT];       // fused weights for this channel

    const int tiles_area = tiles_x * tiles_y;
    const int bc = blockIdx.x / tiles_area;
    const int tile = blockIdx.x - bc * tiles_area;
    const int oh0 = (tile / tiles_x) * TILE_H;
    const int ow0 = (tile % tiles_x) * TILE_W;
    const int c = bc % C;

    const int tid = threadIdx.y * TILE_W + threadIdx.x;

    // ---- stage this channel's fused weights -------------------------------
    for (int i = tid; i < WCOUNT; i += TILE_THREADS) {
        sh_w[i] = __ldg(&fused_w[c * WCOUNT + i]);
    }

    // ---- stage the haloed Haar partial sums -------------------------------
    const T* in_bc = input + (size_t)bc * H * W;
    for (int i = tid; i < SPLANE; i += TILE_THREADS) {
        const int sy = i / SW;
        const int sx = i - sy * SW;
        const int ph = oh0 - R + sy;     // coefficient row this slot holds
        const int pw = ow0 - R + sx;
        float ll = 0.f, lh = 0.f, hl = 0.f, hh = 0.f;
        if (ph >= 0 && ph < H2 && pw >= 0 && pw < W2) {
            const int y0 = 2 * ph, x0 = 2 * pw;
            const T* row0 = in_bc + (size_t)y0 * W + x0;
            const T* row1 = row0 + W;
            haar_step(to_float(__ldg(row0)), to_float(__ldg(row0 + 1)),
                      to_float(__ldg(row1)), to_float(__ldg(row1 + 1)),
                      ll, lh, hl, hh);
        }
        sh_p[0][i] = ll;
        sh_p[1][i] = lh;
        sh_p[2][i] = hl;
        sh_p[3][i] = hh;
    }
    __syncthreads();

    const int h2 = oh0 + threadIdx.y;
    const int w2 = ow0 + threadIdx.x;
    if (h2 >= H2 || w2 >= W2) return;

    // ---- fused conv over the K x K taps -----------------------------------
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    #pragma unroll
    for (int kh = 0; kh < K; ++kh) {
        const int srow = (threadIdx.y + kh) * SW + threadIdx.x;
        const int wrow = kh * K;
        #pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            const int si = srow + kw;
            const int wi = wrow + kw;
            acc0 = fmaf(sh_w[0 * K * K + wi], sh_p[0][si], acc0);
            acc1 = fmaf(sh_w[1 * K * K + wi], sh_p[1][si], acc1);
            acc2 = fmaf(sh_w[2 * K * K + wi], sh_p[2][si], acc2);
            acc3 = fmaf(sh_w[3 * K * K + wi], sh_p[3][si], acc3);
        }
    }

    const int plane = H2 * W2;
    T* out = output + (size_t)bc * 4 * plane + (size_t)h2 * W2 + w2;
    out[0 * plane] = from_float<T>(acc0);
    out[1 * plane] = from_float<T>(acc1);
    out[2 * plane] = from_float<T>(acc2);
    out[3 * plane] = from_float<T>(acc3);

    // Raw (unconvolved) LL for the next decomposition level: the centre tap's
    // partial sum, already in shared memory.
    if (ll_out != nullptr) {
        ll_out[(size_t)bc * plane + (size_t)h2 * W2 + w2] =
            from_float<T>(sh_p[0][(threadIdx.y + R) * SW + threadIdx.x + R]);
    }
}

// -----------------------------------------------------------------------------
// Backward w.r.t. the input.
//
// Same grid as the forward: one thread per output position, which owns exactly
// one 2x2 input block -> writes are exclusive, no atomics. Per subband we first
// accumulate the transposed-conv sum
//     acc[s] = sum_{kh,kw} w_fused[c,s,kh,kw] * grad_out[s, h2-kh+R, w2-kw+R]
// and only then apply the Haar adjoint once (it is tap-independent).
// grad_ll (gradient flowing through the raw-LL output) folds straight into
// acc[LL] with weight 1.
// -----------------------------------------------------------------------------
template<typename T, int K>
__global__ void fused_haar_conv_scale_bwd_kernel(
    const T* __restrict__ grad_output,   // (B, C, 4, H2, W2), contiguous
    const T* __restrict__ grad_ll,       // (B, C, H2, W2) or nullptr
    const float* __restrict__ fused_w,   // (C, 4, K, K)
    T* __restrict__ grad_input,          // (B, C, H, W)
    int C, int H, int W, int H2, int W2,
    int tiles_x, int tiles_y
) {
    constexpr int R = K / 2;
    constexpr int SH = TILE_H + K - 1;
    constexpr int SW = TILE_W + K - 1;
    constexpr int SPLANE = SH * SW;
    constexpr int WCOUNT = 4 * K * K;

    __shared__ float sh_g[4][SPLANE];
    __shared__ float sh_w[WCOUNT];

    const int tiles_area = tiles_x * tiles_y;
    const int bc = blockIdx.x / tiles_area;
    const int tile = blockIdx.x - bc * tiles_area;
    const int oh0 = (tile / tiles_x) * TILE_H;
    const int ow0 = (tile % tiles_x) * TILE_W;
    const int c = bc % C;

    const int tid = threadIdx.y * TILE_W + threadIdx.x;
    const int plane = H2 * W2;

    for (int i = tid; i < WCOUNT; i += TILE_THREADS) {
        sh_w[i] = __ldg(&fused_w[c * WCOUNT + i]);
    }

    const T* go_bc = grad_output + (size_t)bc * 4 * plane;
    for (int i = tid; i < SPLANE; i += TILE_THREADS) {
        const int sy = i / SW;
        const int sx = i - sy * SW;
        const int ph = oh0 - R + sy;
        const int pw = ow0 - R + sx;
        float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
        if (ph >= 0 && ph < H2 && pw >= 0 && pw < W2) {
            const T* p = go_bc + (size_t)ph * W2 + pw;
            g0 = to_float(__ldg(p + 0 * plane));
            g1 = to_float(__ldg(p + 1 * plane));
            g2 = to_float(__ldg(p + 2 * plane));
            g3 = to_float(__ldg(p + 3 * plane));
        }
        sh_g[0][i] = g0;
        sh_g[1][i] = g1;
        sh_g[2][i] = g2;
        sh_g[3][i] = g3;
    }
    __syncthreads();

    const int h2 = oh0 + threadIdx.y;
    const int w2 = ow0 + threadIdx.x;
    if (h2 >= H2 || w2 >= W2) return;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    #pragma unroll
    for (int kh = 0; kh < K; ++kh) {
        // shared row holding grad_output at coefficient row (h2 - kh + R)
        const int srow = (threadIdx.y + (K - 1) - kh) * SW + threadIdx.x + (K - 1);
        const int wrow = kh * K;
        #pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            const int si = srow - kw;
            const int wi = wrow + kw;
            acc0 = fmaf(sh_w[0 * K * K + wi], sh_g[0][si], acc0);
            acc1 = fmaf(sh_w[1 * K * K + wi], sh_g[1][si], acc1);
            acc2 = fmaf(sh_w[2 * K * K + wi], sh_g[2][si], acc2);
            acc3 = fmaf(sh_w[3 * K * K + wi], sh_g[3][si], acc3);
        }
    }

    if (grad_ll != nullptr) {
        acc0 += to_float(__ldg(grad_ll + (size_t)bc * plane + (size_t)h2 * W2 + w2));
    }

    float ga, gb, gc, gd;
    haar_step_adjoint(acc0, acc1, acc2, acc3, ga, gb, gc, gd);

    T* gi = grad_input + (size_t)bc * H * W + (size_t)(2 * h2) * W + 2 * w2;
    gi[0] = from_float<T>(ga);
    gi[1] = from_float<T>(gb);
    gi[W] = from_float<T>(gc);
    gi[W + 1] = from_float<T>(gd);
}

// -----------------------------------------------------------------------------
// Weight gradient, straight from the level input.
//
//   dL/dw_fused[c,s,kh,kw] = sum_{b,h2,w2} grad_out[b,c,s,h2,w2]
//                                        * P_s(h2 + kh - R, w2 + kw - R)
//
// which is the forward's tap loop with the roles of "sum over taps" and "sum
// over positions" swapped. Computing P_s on the fly means the coefficients are
// never materialised here either, so this reads the input and the output
// gradient once each -- against cuDNN's grouped depthwise weight gradient,
// which was the single most expensive kernel in a training step.
//
// Layout: threadIdx.z picks the subband, so a thread only carries K*K
// accumulators in registers (not 4*K*K) and loads one gradient value per
// position; all four subbands share the staged partial-sum tile. Each warp
// reduces its accumulators with shuffles and atomically adds the result, so a
// block's whole spatial chunk costs K*K atomics per warp.
//
// The atomics make this non-deterministic across runs at fp32 rounding level,
// as cuDNN's own weight gradient is.
// -----------------------------------------------------------------------------
template<typename T, int K>
__global__ __launch_bounds__(TILE_W * TILE_H * 4) void fused_haar_grad_weight_kernel(
    const T* __restrict__ input,         // (B, C, H, W), contiguous, even dims
    const T* __restrict__ grad_output,   // (B, C, 4, H2, W2), contiguous
    float* __restrict__ grad_fused_w,    // (C, 4, K, K) fp32, pre-zeroed
    int B, int C, int H, int W, int H2, int W2,
    int tiles_x, int tiles_area
) {
    constexpr int R = K / 2;
    constexpr int SH = TILE_H + K - 1;
    constexpr int SW = TILE_W + K - 1;
    constexpr int SPLANE = SH * SW;
    constexpr int NTHREADS = TILE_W * TILE_H * 4;

    __shared__ float sh_p[4][SPLANE];

    float acc[K * K];
    #pragma unroll
    for (int i = 0; i < K * K; ++i) acc[i] = 0.f;

    const int c = blockIdx.y;
    const int s = threadIdx.z;
    const int tid = (threadIdx.z * TILE_H + threadIdx.y) * TILE_W + threadIdx.x;
    const int plane = H2 * W2;
    const long tiles_total = (long)tiles_area * B;

    for (long tt = blockIdx.x; tt < tiles_total; tt += gridDim.x) {
        const int b = (int)(tt / tiles_area);
        const int tile = (int)(tt - (long)b * tiles_area);
        const int oh0 = (tile / tiles_x) * TILE_H;
        const int ow0 = (tile % tiles_x) * TILE_W;
        const long bc = (long)b * C + c;

        __syncthreads();   // previous iteration's reads must be done
        const T* in_bc = input + (size_t)bc * H * W;
        for (int i = tid; i < SPLANE; i += NTHREADS) {
            const int sy = i / SW;
            const int sx = i - sy * SW;
            const int ph = oh0 - R + sy;
            const int pw = ow0 - R + sx;
            float ll = 0.f, lh = 0.f, hl = 0.f, hh = 0.f;
            if (ph >= 0 && ph < H2 && pw >= 0 && pw < W2) {
                const int y0 = 2 * ph, x0 = 2 * pw;
                const T* row0 = in_bc + (size_t)y0 * W + x0;
                const T* row1 = row0 + W;
                haar_step(to_float(__ldg(row0)), to_float(__ldg(row0 + 1)),
                          to_float(__ldg(row1)), to_float(__ldg(row1 + 1)),
                          ll, lh, hl, hh);
            }
            sh_p[0][i] = ll;
            sh_p[1][i] = lh;
            sh_p[2][i] = hl;
            sh_p[3][i] = hh;
        }
        __syncthreads();

        const int h2 = oh0 + threadIdx.y;
        const int w2 = ow0 + threadIdx.x;
        float g = 0.f;   // out-of-range positions contribute nothing
        if (h2 < H2 && w2 < W2) {
            g = to_float(__ldg(&grad_output[(size_t)bc * 4 * plane + (size_t)s * plane
                                            + (size_t)h2 * W2 + w2]));
        }

        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int srow = (threadIdx.y + kh) * SW + threadIdx.x;
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                acc[kh * K + kw] = fmaf(g, sh_p[s][srow + kw], acc[kh * K + kw]);
            }
        }
    }

    // Warp-reduce each tap and accumulate into the (tiny) global buffer.
    float* out = grad_fused_w + ((size_t)c * 4 + s) * K * K;
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

// -----------------------------------------------------------------------------
// Plain Haar coefficients: (B, C, H, W) -> (B, C, 4, H2, W2).
//
// Used for the weight gradient (which needs the coefficients the conv actually
// saw) and as the gradient of the inverse cascade. Reads through arbitrary
// input strides so it can consume an LL slice without a copy. Odd input
// dimensions are zero padded, matching F.pad in the reference implementation.
// -----------------------------------------------------------------------------
template<typename T>
__global__ void haar_coeffs_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,              // (B, C, 4, H2, W2), contiguous
    int C, int H, int W, int H2, int W2,
    long stride_b, long stride_c, long stride_h, long stride_w,
    long N                               // B * C * H2 * W2
) {
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < N;
         idx += (long)blockDim.x * gridDim.x) {
        const int w2 = idx % W2;
        long t = idx / W2;
        const int h2 = t % H2;
        t /= H2;
        const int c = t % C;
        const int b = t / C;

        const int y0 = 2 * h2, x0 = 2 * w2;
        const T* base = input + b * stride_b + c * stride_c;
        const float a = to_float(base[y0 * stride_h + x0 * stride_w]);
        const float bb = (x0 + 1 < W) ? to_float(base[y0 * stride_h + (x0 + 1) * stride_w]) : 0.f;
        const float cc = (y0 + 1 < H) ? to_float(base[(y0 + 1) * stride_h + x0 * stride_w]) : 0.f;
        const float d = (y0 + 1 < H && x0 + 1 < W)
                        ? to_float(base[(y0 + 1) * stride_h + (x0 + 1) * stride_w]) : 0.f;

        float ll, lh, hl, hh;
        haar_step(a, bb, cc, d, ll, lh, hl, hh);

        const int plane = H2 * W2;
        T* out = output + ((long)b * C + c) * 4 * plane + h2 * W2 + w2;
        out[0 * plane] = from_float<T>(ll);
        out[1 * plane] = from_float<T>(lh);
        out[2 * plane] = from_float<T>(hl);
        out[3 * plane] = from_float<T>(hh);
    }
}

// =============================================================================
// Host wrappers
// =============================================================================

// Only the sizes this build asked for are instantiated; see HAAR_MAX_K.
HAAR_DEFINE_LAUNCHER(fused_haar_conv_scale_kernel)
HAAR_DEFINE_LAUNCHER(fused_haar_conv_scale_bwd_kernel)
HAAR_DEFINE_LAUNCHER(fused_haar_grad_weight_kernel)

// Validates shapes and returns the conv kernel size K taken from the fused
// weight, which is always (C, 4, K, K).
static int check_fused_shapes(const torch::Tensor& x, const torch::Tensor& fused_w,
                              int C, int H, int W) {
    TORCH_CHECK(x.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0,
                "fused Haar conv needs even H, W (pad before calling), got ", H, "x", W);
    TORCH_CHECK(fused_w.is_cuda() && fused_w.is_contiguous(),
                "fused weight must be contiguous CUDA");
    TORCH_CHECK(fused_w.scalar_type() == torch::kFloat32, "fused weight must be float32");
    TORCH_CHECK(fused_w.dim() == 4 && fused_w.size(0) == C && fused_w.size(1) == 4,
                "fused weight must be (C, 4, K, K)");
    const int K = (int)fused_w.size(2);
    TORCH_CHECK(fused_w.size(3) == K, "fused weight must be square");
    // Whether this K was compiled is the dispatch's business, not this check's.
    TORCH_CHECK(K % 2 == 1 && K <= HAAR_K_LIMIT,
                "kernel_size must be odd and <= ", HAAR_K_LIMIT, ", got ", K);
    return K;
}

void fused_haar_conv_forward(
    torch::Tensor input,                       // (B, C, H, W)
    torch::Tensor fused_weight,                // (C, 4, K, K) float32
    torch::Tensor output,                      // (B, C, 4, H2, W2)
    c10::optional<torch::Tensor> ll_output     // (B, C, H2, W2)
) {
    TORCH_CHECK(input.dim() == 4, "input must be (B, C, H, W)");
    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int H2 = H / 2, W2 = W / 2;
    const int K = check_fused_shapes(input, fused_weight, C, H, W);
    TORCH_CHECK(output.is_contiguous() && output.size(3) == H2 && output.size(4) == W2,
                "output must be a contiguous (B, C, 4, H/2, W/2) tensor");

    const int tiles_x = (W2 + TILE_W - 1) / TILE_W;
    const int tiles_y = (H2 + TILE_H - 1) / TILE_H;
    if (tiles_x == 0 || tiles_y == 0) return;

    const long nblocks = (long)tiles_x * tiles_y * B * C;
    TORCH_CHECK(nblocks <= 2147483647L, "grid too large: ", nblocks, " blocks");
    dim3 block(TILE_W, TILE_H);
    dim3 grid((unsigned)nblocks);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* wptr = fused_weight.data_ptr<float>();

    HAAR_DISPATCH_DTYPE(input, "fused_haar_conv_forward", [&] {
        scalar_t* llp = nullptr;
        if (ll_output.has_value()) {
            TORCH_CHECK(ll_output->is_contiguous(), "ll_output must be contiguous");
            llp = haar_ptr<scalar_t>(*ll_output);
        }
        HAAR_DISPATCH_K(fused_haar_conv_scale_kernel, scalar_t,
                        haar_cptr<scalar_t>(input), wptr, haar_ptr<scalar_t>(output),
                        llp, C, H, W, H2, W2, tiles_x, tiles_y);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}

void fused_haar_conv_backward(
    torch::Tensor grad_output,                 // (B, C, 4, H2, W2)
    torch::Tensor fused_weight,                // (C, 4, K, K) float32
    torch::Tensor grad_input,                  // (B, C, H, W)
    c10::optional<torch::Tensor> grad_ll       // (B, C, H2, W2)
) {
    TORCH_CHECK(grad_output.dim() == 5, "grad_output must be (B, C, 4, H2, W2)");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    const int B = grad_input.size(0), C = grad_input.size(1);
    const int H = grad_input.size(2), W = grad_input.size(3);
    const int H2 = H / 2, W2 = W / 2;
    const int K = check_fused_shapes(grad_input, fused_weight, C, H, W);
    TORCH_CHECK(grad_output.size(3) == H2 && grad_output.size(4) == W2,
                "grad_output spatial dims must be H/2, W/2");

    const int tiles_x = (W2 + TILE_W - 1) / TILE_W;
    const int tiles_y = (H2 + TILE_H - 1) / TILE_H;
    if (tiles_x == 0 || tiles_y == 0) return;

    const long nblocks = (long)tiles_x * tiles_y * B * C;
    TORCH_CHECK(nblocks <= 2147483647L, "grid too large: ", nblocks, " blocks");
    dim3 block(TILE_W, TILE_H);
    dim3 grid((unsigned)nblocks);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* wptr = fused_weight.data_ptr<float>();

    HAAR_DISPATCH_DTYPE(grad_output, "fused_haar_conv_backward", [&] {
        const scalar_t* gllp = nullptr;
        if (grad_ll.has_value()) {
            TORCH_CHECK(grad_ll->is_contiguous(), "grad_ll must be contiguous");
            gllp = haar_cptr<scalar_t>(*grad_ll);
        }
        HAAR_DISPATCH_K(fused_haar_conv_scale_bwd_kernel, scalar_t,
                        haar_cptr<scalar_t>(grad_output), gllp, wptr,
                        haar_ptr<scalar_t>(grad_input), C, H, W, H2, W2, tiles_x, tiles_y);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}

void fused_haar_grad_weight(
    torch::Tensor input,                       // (B, C, H, W), even dims
    torch::Tensor grad_output,                 // (B, C, 4, H2, W2)
    torch::Tensor grad_fused_weight            // (C, 4, K, K) float32, zeroed
) {
    TORCH_CHECK(input.dim() == 4, "input must be (B, C, H, W)");
    TORCH_CHECK(grad_output.dim() == 5 && grad_output.is_contiguous(),
                "grad_output must be a contiguous (B, C, 4, H2, W2) tensor");
    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int H2 = H / 2, W2 = W / 2;
    const int K = check_fused_shapes(input, grad_fused_weight, C, H, W);
    TORCH_CHECK(grad_output.size(3) == H2 && grad_output.size(4) == W2,
                "grad_output spatial dims must be H/2, W/2");
    TORCH_CHECK(input.scalar_type() == grad_output.scalar_type(),
                "input and grad_output must share dtype");

    const int tiles_x = (W2 + TILE_W - 1) / TILE_W;
    const int tiles_y = (H2 + TILE_H - 1) / TILE_H;
    const int tiles_area = tiles_x * tiles_y;
    if (tiles_area == 0 || B == 0) return;

    // Aim for a couple of blocks per SM; each block then sweeps a long run of
    // tiles, so the reduction epilogue stays amortised.
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long tiles_total = (long)tiles_area * B;
    long bpc = (2L * sms + C - 1) / C;
    bpc = std::max(1L, std::min(bpc, tiles_total));

    dim3 block(TILE_W, TILE_H, 4);
    dim3 grid((unsigned)bpc, (unsigned)C);
    auto stream = at::cuda::getCurrentCUDAStream();
    float* gwptr = grad_fused_weight.data_ptr<float>();

    HAAR_DISPATCH_DTYPE(input, "fused_haar_grad_weight", [&] {
        HAAR_DISPATCH_K(fused_haar_grad_weight_kernel, scalar_t,
                        haar_cptr<scalar_t>(input), haar_cptr<scalar_t>(grad_output),
                        gwptr, B, C, H, W, H2, W2, tiles_x, tiles_area);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}

void haar_coeffs(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.dim() == 4, "input must be (B, C, H, W)");
    TORCH_CHECK(output.dim() == 5 && output.is_contiguous(),
                "output must be a contiguous (B, C, 4, H2, W2) tensor");
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int H2 = output.size(3), W2 = output.size(4);
    TORCH_CHECK(H2 == (H + 1) / 2 && W2 == (W + 1) / 2,
                "output spatial dims must be ceil(H/2), ceil(W/2)");

    const long N = (long)B * C * H2 * W2;
    if (N == 0) return;
    const int threads = 256;
    const long blocks = std::min<long>((N + threads - 1) / threads, 65535L * 16);
    auto stream = at::cuda::getCurrentCUDAStream();

    HAAR_DISPATCH_DTYPE(input, "haar_coeffs", [&] {
        haar_coeffs_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            haar_cptr<scalar_t>(input), haar_ptr<scalar_t>(output),
            C, H, W, H2, W2,
            input.stride(0), input.stride(1), input.stride(2), input.stride(3), N);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}
