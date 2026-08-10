// =============================================================================
// Fused inverse Haar cascade (1-5 levels) with optional fused addition.
//
// WTConv reconstructs bottom-up: the deepest level's inverse transform is added
// onto the next level's LL subband, which is then inverted, and so on. Doing
// that level by level writes and re-reads every intermediate resolution. This
// kernel walks the whole cascade in registers instead: one thread owns one
// level-1 coefficient position (y1, x1), reads the coefficient it needs at each
// deeper level (position y1 >> (j-1)), and only materialises the final 2x2
// output block.
//
// For intermediate levels only ONE of the four reconstructed values is needed
// (the quadrant this thread descends through), so ihaar_pick replaces a full
// ihaar_step there.
//
// `add` (the base-conv output) is folded into the final store, which removes the
// separate full-resolution add + its extra HBM round trip.
//
// Odd sizes: WTConv zero-pads each level to even before transforming and crops
// the reconstruction back afterwards. A thread only ever reads position
// y1 >> (j-1) < H_j (by induction over the ceil-halving chain), so those crops
// never remove anything the cascade reads; only the final store needs the
// H x W bound check.
// =============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "haar_common.cuh"

template<typename T>
__device__ __forceinline__ void load_subbands(
    const T* __restrict__ data, long offset, int plane, int idx,
    float& ll, float& lh, float& hl, float& hh
) {
    ll = to_float(__ldg(&data[offset + 0 * plane + idx]));
    lh = to_float(__ldg(&data[offset + 1 * plane + idx]));
    hl = to_float(__ldg(&data[offset + 2 * plane + idx]));
    hh = to_float(__ldg(&data[offset + 3 * plane + idx]));
}

template<typename T, int LEVELS, bool HAS_ADD>
__global__ void ihaar_cascade_kernel(
    const T* __restrict__ l1, const T* __restrict__ l2, const T* __restrict__ l3,
    const T* __restrict__ l4, const T* __restrict__ l5,
    const T* __restrict__ add,          // (B, C, H, W) or nullptr
    T* __restrict__ output,             // (B, C, H, W)
    int H, int W,
    int H1, int W1, int H2, int W2, int H3, int W3,
    int H4, int W4, int H5, int W5,
    long N                              // B * C * H1 * W1
) {
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < N;
         idx += (long)blockDim.x * gridDim.x) {
        const int x1 = idx % W1;
        long t = idx / W1;
        const int y1 = t % H1;
        const long bc = t / H1;

        float ll_curr = 0.f;
        float ll, lh, hl, hh;

        if (LEVELS >= 5) {
            const int plane = H5 * W5;
            load_subbands(l5, bc * 4 * plane, plane, (y1 >> 4) * W5 + (x1 >> 4),
                          ll, lh, hl, hh);
            ll_curr = ihaar_pick(ll, lh, hl, hh, (y1 >> 3) & 1, (x1 >> 3) & 1);
        }
        if (LEVELS >= 4) {
            const int plane = H4 * W4;
            load_subbands(l4, bc * 4 * plane, plane, (y1 >> 3) * W4 + (x1 >> 3),
                          ll, lh, hl, hh);
            ll_curr = ihaar_pick(ll + ll_curr, lh, hl, hh, (y1 >> 2) & 1, (x1 >> 2) & 1);
        }
        if (LEVELS >= 3) {
            const int plane = H3 * W3;
            load_subbands(l3, bc * 4 * plane, plane, (y1 >> 2) * W3 + (x1 >> 2),
                          ll, lh, hl, hh);
            ll_curr = ihaar_pick(ll + ll_curr, lh, hl, hh, (y1 >> 1) & 1, (x1 >> 1) & 1);
        }
        if (LEVELS >= 2) {
            const int plane = H2 * W2;
            load_subbands(l2, bc * 4 * plane, plane, (y1 >> 1) * W2 + (x1 >> 1),
                          ll, lh, hl, hh);
            ll_curr = ihaar_pick(ll + ll_curr, lh, hl, hh, y1 & 1, x1 & 1);
        }

        // Level 1: full 2x2 reconstruction
        const int plane1 = H1 * W1;
        load_subbands(l1, bc * 4 * plane1, plane1, y1 * W1 + x1, ll, lh, hl, hh);
        if (LEVELS >= 2) ll += ll_curr;

        float o00, o01, o10, o11;
        ihaar_step(ll, lh, hl, hh, o00, o01, o10, o11);

        const int y = 2 * y1, x = 2 * x1;
        const long out_off = bc * H * W;
        const bool y_ok = (y + 1) < H;
        const bool x_ok = (x + 1) < W;

        if (HAS_ADD) {
            o00 += to_float(__ldg(&add[out_off + (long)y * W + x]));
            if (x_ok) o01 += to_float(__ldg(&add[out_off + (long)y * W + x + 1]));
            if (y_ok) o10 += to_float(__ldg(&add[out_off + (long)(y + 1) * W + x]));
            if (y_ok && x_ok) o11 += to_float(__ldg(&add[out_off + (long)(y + 1) * W + x + 1]));
        }

        if (y < H && x < W) output[out_off + (long)y * W + x] = from_float<T>(o00);
        if (y < H && x_ok)  output[out_off + (long)y * W + x + 1] = from_float<T>(o01);
        if (y_ok && x < W)  output[out_off + (long)(y + 1) * W + x] = from_float<T>(o10);
        if (y_ok && x_ok)   output[out_off + (long)(y + 1) * W + x + 1] = from_float<T>(o11);
    }
}

// =============================================================================
// Host wrapper
// =============================================================================

#define IHAAR_LAUNCH(T, LEVELS, HAS_ADD)                                        \
    ihaar_cascade_kernel<T, LEVELS, HAS_ADD><<<blocks, threads, 0, stream>>>(    \
        p[0], p[1], p[2], p[3], p[4], addp, haar_ptr<T>(output),                 \
        H, W, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], N)

#define IHAAR_DISPATCH_ADD(T, LEVELS)                                           \
    if (addp != nullptr) { IHAAR_LAUNCH(T, LEVELS, true); }                      \
    else                 { IHAAR_LAUNCH(T, LEVELS, false); }

#define IHAAR_DISPATCH_LEVELS(T)                                                \
    switch (num_levels) {                                                        \
        case 1: IHAAR_DISPATCH_ADD(T, 1); break;                                 \
        case 2: IHAAR_DISPATCH_ADD(T, 2); break;                                 \
        case 3: IHAAR_DISPATCH_ADD(T, 3); break;                                 \
        case 4: IHAAR_DISPATCH_ADD(T, 4); break;                                 \
        case 5: IHAAR_DISPATCH_ADD(T, 5); break;                                 \
        default: TORCH_CHECK(false, "ihaar_cascade supports 1-5 levels, got ", num_levels); \
    }

void ihaar_cascade(
    std::vector<torch::Tensor> levels,
    torch::Tensor output,                      // (B, C, H, W)
    c10::optional<torch::Tensor> add
) {
    const int num_levels = (int)levels.size();
    TORCH_CHECK(num_levels >= 1 && num_levels <= 5,
                "ihaar_cascade supports 1-5 levels, got ", num_levels);
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "output must be contiguous CUDA");

    const int B = (int)levels[0].size(0), C = (int)levels[0].size(1);
    const int H1 = (int)levels[0].size(3), W1 = (int)levels[0].size(4);
    const int H = (int)output.size(2), W = (int)output.size(3);

    int d[10];
    for (int i = 0; i < 5; ++i) { d[2 * i] = 1; d[2 * i + 1] = 1; }
    for (int i = 0; i < num_levels; ++i) {
        const auto& l = levels[i];
        TORCH_CHECK(l.dim() == 5 && l.size(2) == 4, "level ", i, " must be (B, C, 4, H, W)");
        TORCH_CHECK(l.is_contiguous(), "level ", i, " must be contiguous");
        TORCH_CHECK(l.size(0) == B && l.size(1) == C, "levels must share B and C");
        TORCH_CHECK(l.scalar_type() == output.scalar_type(), "levels and output must share dtype");
        d[2 * i] = (int)l.size(3);
        d[2 * i + 1] = (int)l.size(4);
    }
    TORCH_CHECK(H <= 2 * H1 && W <= 2 * W1, "output is larger than level 1 can reconstruct");

    const long N = (long)B * C * H1 * W1;
    if (N == 0) return;
    const int threads = 256;
    const long blocks = std::min<long>((N + threads - 1) / threads, 65535L * 16);
    auto stream = at::cuda::getCurrentCUDAStream();

    HAAR_DISPATCH_DTYPE(output, "ihaar_cascade", [&] {
        const scalar_t* p[5];
        for (int i = 0; i < 5; ++i) {
            p[i] = haar_cptr<scalar_t>(levels[i < num_levels ? i : 0]);
        }
        const scalar_t* addp = nullptr;
        if (add.has_value()) {
            TORCH_CHECK(add->is_contiguous() && add->sizes() == output.sizes(),
                        "add tensor must be contiguous and match the output shape");
            TORCH_CHECK(add->scalar_type() == output.scalar_type(),
                        "add tensor must share the output dtype");
            addp = haar_cptr<scalar_t>(*add);
        }
        IHAAR_DISPATCH_LEVELS(scalar_t);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}
