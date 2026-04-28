#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>

// ============================================================
// SM90 bwd_intra builder helpers
//
// Mirror SM100 util_func.h `setup_*_intra` / `setup_*_intra_2gn`
// structurally:
//   * float4 packed gmem/smem loads
//   * uint2 bf16-pack unpack
//   * 4-wide stores into B-operand smem
//
// SM90 adaptations vs SM100 helpers:
//   * No `add.f32x2` PTX (sm_100-only) — emit scalar fp32 ops instead.
//     Compiler vectorizes when registers are paired, so generated SASS is
//     equivalent to the SM100 path on Hopper.
//   * `nvbf16x4` → `uint2`-bitcast already used on SM90.
//   * Threading: 128 threads tile (BC=16 rows × BK=32 cols / 4-elem chunk
//     → 128 chunks → 1 chunk/thread).
//
// All helpers follow this contract:
//   * Caller does the surrounding `__syncthreads()` / `cp.async.wait_group`.
//   * Builders write to a B-operand smem tensor with logical shape (BK=32, BC=16),
//     which is what the MMA partitioner expects.
// ============================================================

namespace sm90_bwd_intra {

using namespace cute;

__device__ __forceinline__ __nv_bfloat162 _bitcast_bf162(uint32_t x) {
    return reinterpret_cast<__nv_bfloat162&>(x);
}

// ──────────────────────────────────────────────────────────────────
// Phase 1 off-diagonal builder (reads gmem K, G):
//   KG[n, k_dim] = K[k_dim, n] * exp2f(gn[n] - G[k_dim, n])
// Mirrors sm100 setup_kg_intra (1 SMEM/GMEM load → 4-wide store).
// 1 thread = 1 (k_dim, n..n+3) chunk.
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS, class GK, class GG, class SKG>
__device__ __forceinline__ void
setup_kg_intra_offdiag_gmem(
    SKG& sKG, GK const& gK_j, GG const& gG_j,
    const float* s_gn, int tid)
{
    constexpr int VEC_ELEMS = BC * BK / 4;
    static_assert(VEC_ELEMS == NUM_THREADS, "1 chunk/thread expected");
    int vi = tid;
    int k_dim = (vi * 4) / BK;
    int n     = (vi * 4) % BK;

    uint2 k_pack = *reinterpret_cast<const uint2*>(&gK_j(k_dim, n));
    __nv_bfloat162 k01 = _bitcast_bf162(k_pack.x);
    __nv_bfloat162 k23 = _bitcast_bf162(k_pack.y);
    float2 kf01 = __bfloat1622float2(k01);
    float2 kf23 = __bfloat1622float2(k23);
    float4 gv  = *reinterpret_cast<const float4*>(&gG_j(k_dim, n));
    float4 gnv = *reinterpret_cast<const float4*>(&s_gn[n]);

    sKG(n + 0, k_dim) = kf01.x * exp2f(gnv.x - gv.x);
    sKG(n + 1, k_dim) = kf01.y * exp2f(gnv.y - gv.y);
    sKG(n + 2, k_dim) = kf23.x * exp2f(gnv.z - gv.z);
    sKG(n + 3, k_dim) = kf23.y * exp2f(gnv.w - gv.w);
}

// ──────────────────────────────────────────────────────────────────
// Phase 1 diagonal builder (reads persistent smem K, G):
//   KG[n, k_dim] = K[k_dim, n] * exp2f(-(G[k_dim, n] - gn[n]))
// 1 thread = 1 element (BK*BC=512 elems / 128 threads = 4 elems/thread).
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS, class SK, class SG, class SKG>
__device__ __forceinline__ void
setup_kg_intra_diag(
    SKG& sKG, SK const& sK, SG const& sG,
    const float* s_gn, int T_seq, int i_ti, int tid)
{
    for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
        int n = idx % BK;
        int k_dim = idx / BK;
        bool valid = (i_ti + k_dim) < T_seq;
        float g_diff = valid ? (sG(k_dim, n) - s_gn[n]) : 0.f;
        sKG(n, k_dim) = valid ? (__bfloat162float(sK(k_dim, n)) * exp2f(-g_diff)) : 0.f;
    }
}

// ──────────────────────────────────────────────────────────────────
// Phase 2 off-diagonal fused builder (reads gmem Q, K, G, beta):
//   QG [n, k_dim] = Q[k_dim, n] * exp2f(G[k_dim, n] - gn[n])
//   KBG[n, k_dim] = K[k_dim, n] * beta[k_dim] * exp2f(G[k_dim, n] - gn[n])
// Mirrors sm100 setup_intra_fused (single G/K/Q load → 2 outputs).
// 1 thread = 1 (k_dim, n..n+3) chunk.
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS,
          class GQ, class GK, class GG, class GBETA, class SQG, class SKBG>
__device__ __forceinline__ void
setup_intra_fused_offdiag_gmem(
    SQG& sQG, SKBG& sKBG,
    GQ const& gQ_j, GK const& gK_j, GG const& gG_j, GBETA const& gBeta_j,
    const float* s_gn, int T_seq, int j_ti, int tid)
{
    constexpr int VEC_ELEMS = BC * BK / 4;
    static_assert(VEC_ELEMS == NUM_THREADS, "1 chunk/thread expected");
    int vi = tid;
    int k_dim = (vi * 4) / BK;
    int n     = (vi * 4) % BK;
    bool valid = (j_ti + k_dim) < T_seq;

    if (valid) {
        float bv = __bfloat162float(gBeta_j(k_dim));

        uint2 q_pack = *reinterpret_cast<const uint2*>(&gQ_j(k_dim, n));
        uint2 k_pack = *reinterpret_cast<const uint2*>(&gK_j(k_dim, n));
        __nv_bfloat162 q01 = _bitcast_bf162(q_pack.x);
        __nv_bfloat162 q23 = _bitcast_bf162(q_pack.y);
        __nv_bfloat162 k01 = _bitcast_bf162(k_pack.x);
        __nv_bfloat162 k23 = _bitcast_bf162(k_pack.y);
        float2 qf01 = __bfloat1622float2(q01);
        float2 qf23 = __bfloat1622float2(q23);
        float2 kf01 = __bfloat1622float2(k01);
        float2 kf23 = __bfloat1622float2(k23);
        float4 gv  = *reinterpret_cast<const float4*>(&gG_j(k_dim, n));
        float4 gnv = *reinterpret_cast<const float4*>(&s_gn[n]);

        float gate0 = exp2f(gv.x - gnv.x);
        float gate1 = exp2f(gv.y - gnv.y);
        float gate2 = exp2f(gv.z - gnv.z);
        float gate3 = exp2f(gv.w - gnv.w);

        sQG(n + 0, k_dim) = qf01.x * gate0;
        sQG(n + 1, k_dim) = qf01.y * gate1;
        sQG(n + 2, k_dim) = qf23.x * gate2;
        sQG(n + 3, k_dim) = qf23.y * gate3;

        sKBG(n + 0, k_dim) = kf01.x * bv * gate0;
        sKBG(n + 1, k_dim) = kf01.y * bv * gate1;
        sKBG(n + 2, k_dim) = kf23.x * bv * gate2;
        sKBG(n + 3, k_dim) = kf23.y * bv * gate3;
    } else {
        sQG(n + 0, k_dim) = 0.f;
        sQG(n + 1, k_dim) = 0.f;
        sQG(n + 2, k_dim) = 0.f;
        sQG(n + 3, k_dim) = 0.f;
        sKBG(n + 0, k_dim) = 0.f;
        sKBG(n + 1, k_dim) = 0.f;
        sKBG(n + 2, k_dim) = 0.f;
        sKBG(n + 3, k_dim) = 0.f;
    }
}

// ──────────────────────────────────────────────────────────────────
// Phase 2 diagonal fused builder (reads persistent smem Q, K, G):
//   QG [n, k_dim] = Q[k_dim, n] * exp2f(G[k_dim, n] - gn[n])
//   KBG[n, k_dim] = K[k_dim, n] * beta[k_dim] * exp2f(G[k_dim, n] - gn[n])
// 1 thread = 1 element.
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS,
          class SQ, class SK, class SG, class SQG, class SKBG>
__device__ __forceinline__ void
setup_intra_fused_diag(
    SQG& sQG, SKBG& sKBG,
    SQ const& sQ, SK const& sK, SG const& sG,
    const float* s_beta, const float* s_gn,
    int T_seq, int i_ti, int tid)
{
    for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
        int n = idx % BK;
        int k_dim = idx / BK;
        bool valid = (i_ti + k_dim) < T_seq;
        float g_diff = valid ? (sG(k_dim, n) - s_gn[n]) : 0.f;
        float exp_g = valid ? exp2f(g_diff) : 0.f;
        sQG(n, k_dim) = valid ? __bfloat162float(sQ(k_dim, n)) * exp_g : 0.f;
        float kv = valid ? __bfloat162float(sK(k_dim, n)) : 0.f;
        sKBG(n, k_dim) = kv * s_beta[k_dim] * exp_g;
    }
}

}  // namespace sm90_bwd_intra
