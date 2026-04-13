#pragma once

#include <cute/tensor.hpp>

#include "kda_bwd_basic.h"

namespace sm90_bwd {

using namespace cute;

// =====================================================================
// float2 operations — direct copy from SM100
// =====================================================================

CUTE_DEVICE
float2
float2_add(const float2& a, const float2& b) {
    float2 c;
    asm volatile("add.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2
float2_sub(const float2& a, const float2& b) {
    float2 c;
    asm volatile("sub.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2
float2_mul(const float2& a, const float2& b) {
    float2 c;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2
float2_fma(const float2& a, const float2& b, const float2& c) {
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)),
                   "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

CUTE_DEVICE
__nv_bfloat162
bf16x2_add(const __nv_bfloat162& a, const __nv_bfloat162& b) {
    __nv_bfloat162 c;
    asm volatile("add.bf16x2 %0, %1, %2;\n"
                 : "=r"(reinterpret_cast<uint32_t&>(c))
                 : "r"(reinterpret_cast<uint32_t const&>(a)), "r"(reinterpret_cast<uint32_t const&>(b)));
    return c;
}

// =====================================================================
// store_256B — efficient global memory write (256 bytes = 8 x float4)
// =====================================================================

CUTE_DEVICE
void
_store_256B(
    uint32_t const& src0,
    uint32_t const& src1,
    uint32_t const& src2,
    uint32_t const& src3,
    uint32_t const& src4,
    uint32_t const& src5,
    uint32_t const& src6,
    uint32_t const& src7,
    void* gmem_addr) {
    asm volatile(
        "st.global.L1::no_allocate.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n" ::"l"(gmem_addr),
        "r"(src0),
        "r"(src1),
        "r"(src2),
        "r"(src3),
        "r"(src4),
        "r"(src5),
        "r"(src6),
        "r"(src7));
}

CUTE_DEVICE
void
store_256B(void* src, void* dst) {
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src);
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
    _store_256B(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], src_ptr[7], dst);
}

// =====================================================================
// launch_tma_copy — TMA async copy helper
// =====================================================================

template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void
launch_tma_copy(
    const TMA& tma_copy,
    Tensor0 src,
    Tensor1 dst,
    uint64_t& bar,
    const cute::TMA::CacheHintSm90& cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    auto thr_tma = tma_copy.get_slice(_0{});
    cute::copy(tma_copy.with(bar, 0, cache_hint), thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

// =====================================================================
// store_128b — 128-bit store helper
// =====================================================================

template <typename T>
CUTE_DEVICE void
store_128b(void* smem_ptr, const T& data) {
    static_assert(sizeof(T) == 16);
    *(__int128*)smem_ptr = *(__int128*)&data;
}

}  // namespace sm90_bwd
