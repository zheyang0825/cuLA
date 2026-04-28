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

// ============================================================
// SM90 bwd_intra epilogue helpers
//
// Mirror SM100 util_func.h `epilogue_output_dq/dg/dk` structurally:
//   * named helpers per output tensor
//   * preload-then-scatter-add-then-store flow
//
// SM90 adaptations vs SM100:
//   * SM100 has TMEM (1 thread = 1 row × K_TILE cols), can write 256B chunks
//     directly to gmem. SM90's MMA accumulator is scattered registers
//     (row = lane/4 + (v/2)*8, col = lane%4*2 + warp*8), so we MUST stage
//     through smem to coalesce. The "epilogue" pattern below preserves
//     coalesced gmem stores via a single sStage scratchpad shared with
//     dg-computation.
//   * Per-thread "pair" accessor: 2 fp32 values at adjacent cols
//     (v=2*rh, v=2*rh+1) → float2 LDS/STS replaces 4 scalar scatters.
// ============================================================

namespace sm90_bwd_intra {

// SM80 16x8x8 f32 accumulator scatter mapping.
__device__ __forceinline__ void
get_acc_row_col(int tid, int v, int& row, int& col) {
    int lane = tid % 32;
    int warp_id = tid / 32;
    row = (lane / 4) + (v / 2) * 8;
    col = (lane % 4) * 2 + (v % 2) + warp_id * 8;
}

// Per-thread "pair" base position (rh in {0,1}): two adjacent fp32 acc values
// (v=2*rh, v=2*rh+1) live at (row, col_base..col_base+1).
__device__ __forceinline__ void
get_acc_pair_row_col(int tid, int rh, int& row, int& col_base) {
    int lane = tid % 32;
    int warp_id = tid / 32;
    row = (lane / 4) + rh * 8;
    col_base = (lane % 4) * 2 + warp_id * 8;
}

// ──────────────────────────────────────────────────────────────────
// epilogue_output_dq: preload dq_prev → sStage, scatter-add dq_acc,
//                     vectorized bf16x4 store to gmem.
// Caller must __syncthreads() before and after this helper.
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS, class GDq, class GDqOut, class SStage>
__device__ __forceinline__ void
epilogue_output_dq(
    GDq const& gDq_tile, GDqOut& gDqOut_tile, SStage& sStage,
    const float dq_acc[4], int T_seq, int i_ti, int tid)
{
    static_assert(BC * BK / 4 == NUM_THREADS, "1 chunk/thread expected");
    // Preload dq_prev row-major.
    int vi = tid;
    int r = (vi * 4) / BK;
    int c = (vi * 4) % BK;
    float4 prev = {0.f, 0.f, 0.f, 0.f};
    if ((i_ti + r) < T_seq) {
        prev = *reinterpret_cast<const float4*>(&gDq_tile(r, c));
    }
    sStage(r, c + 0) = prev.x;
    sStage(r, c + 1) = prev.y;
    sStage(r, c + 2) = prev.z;
    sStage(r, c + 3) = prev.w;
    __syncthreads();

    // Scatter add via per-thread (row, col_base) accessor.
    for (int rh = 0; rh < 2; ++rh) {
        int row, col_base;
        get_acc_pair_row_col(tid, rh, row, col_base);
        float2* p = reinterpret_cast<float2*>(&sStage(row, col_base));
        float2 cur = *p;
        cur.x += dq_acc[rh * 2 + 0];
        cur.y += dq_acc[rh * 2 + 1];
        *p = cur;
    }
    __syncthreads();

    // Vectorized bf16x4 store (64-bit per thread).
    if ((i_ti + r) < T_seq) {
        __nv_bfloat162 lo = __floats2bfloat162_rn(sStage(r, c + 0), sStage(r, c + 1));
        __nv_bfloat162 hi = __floats2bfloat162_rn(sStage(r, c + 2), sStage(r, c + 3));
        uint2 packed;
        packed.x = reinterpret_cast<uint32_t&>(lo);
        packed.y = reinterpret_cast<uint32_t&>(hi);
        *reinterpret_cast<uint2*>(&gDqOut_tile(r, c)) = packed;
    }
}

// ──────────────────────────────────────────────────────────────────
// epilogue_output_dg: scatter-compute dg = q*dq + (dk2-dkt)*k + dg_prev
//                     into sStage, vectorized float4 store to gmem.
// Caller must __syncthreads() before and after this helper.
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS,
          class SQ, class SK, class GDg, class GDgOut, class SStage>
__device__ __forceinline__ void
epilogue_output_dg(
    SQ const& sQ, SK const& sK,
    GDg const& gDg_tile, GDgOut& gDgOut_tile, SStage& sStage,
    const float dq_acc[4], const float dk_acc[4], const float dkt_acc[4],
    int T_seq, int i_ti, bool is_boundary, int tid)
{
    static_assert(BC * BK / 4 == NUM_THREADS, "1 chunk/thread expected");
    for (int rh = 0; rh < 2; ++rh) {
        int row, col_base;
        get_acc_pair_row_col(tid, rh, row, col_base);
        uint32_t qpack = *reinterpret_cast<uint32_t*>(&sQ(row, col_base));
        uint32_t kpack = *reinterpret_cast<uint32_t*>(&sK(row, col_base));
        __nv_bfloat162 q2 = _bitcast_bf162(qpack);
        __nv_bfloat162 k2 = _bitcast_bf162(kpack);
        float2 qf = __bfloat1622float2(q2);
        float2 kf = __bfloat1622float2(k2);
        float2 prev{0.f, 0.f};
        if (!is_boundary || (i_ti + row) < T_seq) {
            prev = *reinterpret_cast<const float2*>(&gDg_tile(row, col_base));
        }
        float2 out;
        out.x = qf.x * dq_acc[rh * 2 + 0] + (dk_acc[rh * 2 + 0] - dkt_acc[rh * 2 + 0]) * kf.x + prev.x;
        out.y = qf.y * dq_acc[rh * 2 + 1] + (dk_acc[rh * 2 + 1] - dkt_acc[rh * 2 + 1]) * kf.y + prev.y;
        *reinterpret_cast<float2*>(&sStage(row, col_base)) = out;
    }
    __syncthreads();

    int vi = tid;
    int r = (vi * 4) / BK;
    int c = (vi * 4) % BK;
    if ((i_ti + r) < T_seq) {
        float4 val;
        val.x = sStage(r, c + 0);
        val.y = sStage(r, c + 1);
        val.z = sStage(r, c + 2);
        val.w = sStage(r, c + 3);
        *reinterpret_cast<float4*>(&gDgOut_tile(r, c)) = val;
    }
}

// ──────────────────────────────────────────────────────────────────
// epilogue_output_dk: preload dk_prev → sStage, scatter-add dk_acc + dkt_acc,
//                     vectorized bf16x4 store to gmem.
// Caller must __syncthreads() before and after this helper.
// ──────────────────────────────────────────────────────────────────
template <int BC, int BK, int NUM_THREADS, class GDk, class GDkOut, class SStage>
__device__ __forceinline__ void
epilogue_output_dk(
    GDk const& gDk_tile, GDkOut& gDkOut_tile, SStage& sStage,
    const float dk_acc[4], const float dkt_acc[4],
    int T_seq, int i_ti, int tid)
{
    static_assert(BC * BK / 4 == NUM_THREADS, "1 chunk/thread expected");
    int vi = tid;
    int r = (vi * 4) / BK;
    int c = (vi * 4) % BK;
    float4 prev = {0.f, 0.f, 0.f, 0.f};
    if ((i_ti + r) < T_seq) {
        prev = *reinterpret_cast<const float4*>(&gDk_tile(r, c));
    }
    sStage(r, c + 0) = prev.x;
    sStage(r, c + 1) = prev.y;
    sStage(r, c + 2) = prev.z;
    sStage(r, c + 3) = prev.w;
    __syncthreads();

    for (int rh = 0; rh < 2; ++rh) {
        int row, col_base;
        get_acc_pair_row_col(tid, rh, row, col_base);
        float2* p = reinterpret_cast<float2*>(&sStage(row, col_base));
        float2 cur = *p;
        cur.x += dk_acc[rh * 2 + 0] + dkt_acc[rh * 2 + 0];
        cur.y += dk_acc[rh * 2 + 1] + dkt_acc[rh * 2 + 1];
        *p = cur;
    }
    __syncthreads();

    if ((i_ti + r) < T_seq) {
        __nv_bfloat162 lo = __floats2bfloat162_rn(sStage(r, c + 0), sStage(r, c + 1));
        __nv_bfloat162 hi = __floats2bfloat162_rn(sStage(r, c + 2), sStage(r, c + 3));
        uint2 packed;
        packed.x = reinterpret_cast<uint32_t&>(lo);
        packed.y = reinterpret_cast<uint32_t&>(hi);
        *reinterpret_cast<uint2*>(&gDkOut_tile(r, c)) = packed;
    }
}

}  // namespace sm90_bwd_intra

