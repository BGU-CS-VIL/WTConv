// =============================================================================
// Shared device helpers for the fused Haar CUDA kernels.
//
// Subband convention (matches WTConv's db1 filters exactly). For a 2x2 input
// block
//     a = x[2h  , 2w  ]   b = x[2h  , 2w+1]
//     c = x[2h+1, 2w  ]   d = x[2h+1, 2w+1]
// the four coefficients are
//     LL = 0.5*(a + b + c + d)
//     LH = 0.5*(a + b - c - d)
//     HL = 0.5*(a - b + c - d)
//     HH = 0.5*(a - b - c + d)
// The 4x4 matrix formed by those rows is orthogonal, so the inverse transform
// is its transpose (see ihaar_step / ihaar_pick below).
// =============================================================================

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Largest convolution kernel size these kernels can support at all. This is a
// hard ceiling, not a build option: the forward/grad-input kernels stage
// 16*(2K^2 + 38K + 217) bytes of static shared memory, which is 48016 B at
// K = 29 and 53072 B at K = 31 -- past the 48 KiB per-block static limit. Those
// compile fine and then fail at launch with cudaErrorInvalidArgument, so the
// dispatch must not instantiate beyond 29.
#define HAAR_K_LIMIT 29

// Largest kernel size actually instantiated by this build. Every odd K up to
// this value is compiled for all three dtypes, and each instance is expensive:
// the weight-gradient kernels hold K*K accumulators in a fully unrolled register
// array, so compile time grows superlinearly in K. Building the whole range up
// to HAAR_K_LIMIT takes minutes, and most models only ever use K <= 7, so the
// default stops there and larger K is an opt-in rebuild:
//
//     HAAR_MAX_K=29 python your_script.py
//
// Requesting a K above what was built is an error, not a slow fallback; the
// message names this knob. Each value gets its own cached build (see the
// extension name in haar_cuda.py), so switching back and forth is free.
#ifndef HAAR_MAX_K
#define HAAR_MAX_K 7
#endif

#if (HAAR_MAX_K) > (HAAR_K_LIMIT) || (HAAR_MAX_K) < 1 || (HAAR_MAX_K) % 2 == 0
#error "HAAR_MAX_K must be odd, >= 1 and <= 29"
#endif

// -----------------------------------------------------------------------------
// Type conversion helpers (compute always happens in fp32)
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
// Forward Haar on one 2x2 block
// -----------------------------------------------------------------------------
__device__ __forceinline__ void haar_step(
    float a, float b, float c, float d,
    float& ll, float& lh, float& hl, float& hh
) {
    const float sum_ac = a + c, sum_bd = b + d;
    const float diff_ac = a - c, diff_bd = b - d;
    ll = 0.5f * (sum_ac + sum_bd);
    lh = 0.5f * (diff_ac + diff_bd);
    hl = 0.5f * (sum_ac - sum_bd);
    hh = 0.5f * (diff_ac - diff_bd);
}

// -----------------------------------------------------------------------------
// Inverse Haar: reconstruct a full 2x2 block from 4 coefficients
// -----------------------------------------------------------------------------
__device__ __forceinline__ void ihaar_step(
    float ll, float lh, float hl, float hh,
    float& a, float& b, float& c, float& d
) {
    const float ll_p_lh = ll + lh, ll_m_lh = ll - lh;
    const float hl_p_hh = hl + hh, hl_m_hh = hl - hh;
    a = 0.5f * (ll_p_lh + hl_p_hh);
    b = 0.5f * (ll_p_lh - hl_p_hh);
    c = 0.5f * (ll_m_lh + hl_m_hh);
    d = 0.5f * (ll_m_lh - hl_m_hh);
}

// -----------------------------------------------------------------------------
// Inverse Haar, single quadrant (qy, qx) of the reconstructed 2x2 block.
// Used by the cascade, where intermediate levels only need one of the four.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float ihaar_pick(
    float ll, float lh, float hl, float hh, int qy, int qx
) {
    const float lh_s = qy ? -lh : lh;
    const float hh_s = qy ? -hh : hh;
    const float t1 = ll + lh_s;
    float t2 = hl + hh_s;
    if (qx) t2 = -t2;
    return 0.5f * (t1 + t2);
}

// -----------------------------------------------------------------------------
// Adjoint of the forward Haar: spread 4 coefficient gradients back onto the
// 2x2 input block. Identical algebra to ihaar_step (the transform is
// orthogonal) but kept separate for readability at the call sites.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void haar_step_adjoint(
    float g_ll, float g_lh, float g_hl, float g_hh,
    float& ga, float& gb, float& gc, float& gd
) {
    const float ll_p_lh = g_ll + g_lh, ll_m_lh = g_ll - g_lh;
    const float hl_p_hh = g_hl + g_hh, hl_m_hh = g_hl - g_hh;
    ga = 0.5f * (ll_p_lh + hl_p_hh);
    gb = 0.5f * (ll_p_lh - hl_p_hh);
    gc = 0.5f * (ll_m_lh + hl_m_hh);
    gd = 0.5f * (ll_m_lh - hl_m_hh);
}

// -----------------------------------------------------------------------------
// dtype dispatch helper
// -----------------------------------------------------------------------------
#define HAAR_DISPATCH_DTYPE(TENSOR, NAME, ...)                                  \
    [&] {                                                                       \
        const auto _st = (TENSOR).scalar_type();                                \
        if (_st == torch::kFloat32)      { using scalar_t = float;         __VA_ARGS__(); } \
        else if (_st == torch::kFloat16) { using scalar_t = __half;        __VA_ARGS__(); } \
        else if (_st == torch::kBFloat16){ using scalar_t = __nv_bfloat16; __VA_ARGS__(); } \
        else TORCH_CHECK(false, NAME ": unsupported dtype ", _st,                \
                         " (supported: float32, float16, bfloat16)");           \
    }()

// -----------------------------------------------------------------------------
// Kernel-size dispatch
// -----------------------------------------------------------------------------
// Defines haar_launcher_<KERNEL>, a launcher whose disabled specialisation never
// names the kernel, so nvcc only instantiates the sizes this build asked for.
//
// A plain `if constexpr (K <= HAAR_MAX_K)` at the call site would not work: the
// dispatch switches live in ordinary (non-template) host functions, and there a
// discarded constexpr branch still instantiates everything it names -- which is
// the whole cost we are avoiding. Selecting between two class-template
// specialisations does prevent the instantiation.
#define HAAR_DEFINE_LAUNCHER(KERNEL)                                            \
    template<typename T, int K, bool Enabled = (K <= HAAR_MAX_K)>               \
    struct haar_launcher_##KERNEL {                                             \
        template<typename... Args>                                              \
        static void run(dim3 grid, dim3 block, cudaStream_t stream, Args... a) { \
            KERNEL<T, K><<<grid, block, 0, stream>>>(a...);                     \
        }                                                                       \
    };                                                                          \
    template<typename T, int K>                                                 \
    struct haar_launcher_##KERNEL<T, K, false> {                                \
        template<typename... Args>                                              \
        static void run(dim3, dim3, cudaStream_t, Args...) {                    \
            TORCH_CHECK(false, "kernel_size ", K, " was not compiled into this " \
                        "build (built with HAAR_MAX_K=", HAAR_MAX_K,            \
                        "); rebuild with HAAR_MAX_K=", K);                      \
        }                                                                       \
    };

// Launch KERNEL<T, K> for a compile-time K, going through its launcher. Reads
// grid, block and stream from the enclosing scope, as the host wrappers name them.
#define HAAR_LAUNCH_K(KERNEL, T, K, ...)                                        \
    haar_launcher_##KERNEL<T, K>::run(grid, block, stream, __VA_ARGS__)

// Every odd kernel size up to HAAR_K_LIMIT gets a case; the ones this build
// skipped resolve to the disabled launcher and report that at runtime.
#define HAAR_DISPATCH_K(KERNEL, T, ...)                                         \
    switch (K) {                                                                \
        case  1: HAAR_LAUNCH_K(KERNEL, T,  1, __VA_ARGS__); break;              \
        case  3: HAAR_LAUNCH_K(KERNEL, T,  3, __VA_ARGS__); break;              \
        case  5: HAAR_LAUNCH_K(KERNEL, T,  5, __VA_ARGS__); break;              \
        case  7: HAAR_LAUNCH_K(KERNEL, T,  7, __VA_ARGS__); break;              \
        case  9: HAAR_LAUNCH_K(KERNEL, T,  9, __VA_ARGS__); break;              \
        case 11: HAAR_LAUNCH_K(KERNEL, T, 11, __VA_ARGS__); break;              \
        case 13: HAAR_LAUNCH_K(KERNEL, T, 13, __VA_ARGS__); break;              \
        case 15: HAAR_LAUNCH_K(KERNEL, T, 15, __VA_ARGS__); break;              \
        case 17: HAAR_LAUNCH_K(KERNEL, T, 17, __VA_ARGS__); break;              \
        case 19: HAAR_LAUNCH_K(KERNEL, T, 19, __VA_ARGS__); break;              \
        case 21: HAAR_LAUNCH_K(KERNEL, T, 21, __VA_ARGS__); break;              \
        case 23: HAAR_LAUNCH_K(KERNEL, T, 23, __VA_ARGS__); break;              \
        case 25: HAAR_LAUNCH_K(KERNEL, T, 25, __VA_ARGS__); break;              \
        case 27: HAAR_LAUNCH_K(KERNEL, T, 27, __VA_ARGS__); break;              \
        case 29: HAAR_LAUNCH_K(KERNEL, T, 29, __VA_ARGS__); break;              \
        default: TORCH_CHECK(false, "kernel_size must be odd and <= ",          \
                             HAAR_K_LIMIT, ", got ", K);                        \
    }

// Raw typed pointer from a tensor, for the three supported dtypes.
template<typename T>
inline const T* haar_cptr(const torch::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}
template<typename T>
inline T* haar_ptr(torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}
