// SM90 KDA Backward Intra-Chunk Kernel — Phase 1: Framework + Infrastructure
//
// Design: 416 threads = MMA WG + Prep×2 + Load warp
// MMA: SM80 mma.sync TF32 sub-chunk loop (16×8×8)
// B operands: fp32 row-major (hardware truncation to TF32)
// Causal mask: on-the-fly in registers during A operand loading

#include <cute/arch/mma_sm80.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>

#include "kda_bwd_helpers.h"
#include "kda_bwd_intra_sm90.cuh"
#include "kda_bwd_setup_func.h"
#include "kda_bwd_utils.h"

// Debug print switch (gated to one tile / one warp to limit volume)
#ifndef KDA_BWD_SM90_DEBUG_PRINT
#define KDA_BWD_SM90_DEBUG_PRINT 0
#endif

namespace sm90_bwd {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using namespace cute;

// =====================================================================
// Constants (aligned with design.md)
// =====================================================================
constexpr int SUB_T_TILE = 16;                // sub-chunk size
constexpr int T_TILE = 64;                    // chunk size (tokens per tile)
constexpr int K_SIZE = 128;                   // head dimension
constexpr int K_TILE = 32;                    // per-ki tile width
constexpr int K_ITERATION = K_SIZE / K_TILE;  // = 4
constexpr int NUM_BUF = 2;                      // value buffers (Q/K/G/dQ/dK/dG): double-buffered
constexpr int NUM_BUF_A = 1;                    // dA/beta buffers: single-buffered (SMEM-limited)
constexpr int NUM_MMA_THREADS = 128;
constexpr int NUM_PREP_THREADS = 256;
constexpr int NUM_LOAD_THREADS = 32;
constexpr int MMA_THREAD_OFFSET = 0;
constexpr int PREP_THREAD_OFFSET = MMA_THREAD_OFFSET + NUM_MMA_THREADS;
constexpr int LOAD_THREAD_OFFSET = PREP_THREAD_OFFSET + NUM_PREP_THREADS;
constexpr int NUM_THREADS = NUM_MMA_THREADS + NUM_PREP_THREADS + NUM_LOAD_THREADS;
constexpr int NUM_MMA_WARPS = NUM_MMA_THREADS / 32;
constexpr int NUM_PREP_WARPS = NUM_PREP_THREADS / 32;
constexpr int LOAD_WARP_IDX = NUM_MMA_WARPS + NUM_PREP_WARPS;
constexpr int CHUNK_SIZE = 64;

// Clamped exp2f to prevent inf/NaN from large gate differences across sub-tiles
__device__ __forceinline__ float
safe_exp2f(float x) {
    return exp2f(fminf(fmaxf(x, -126.0f), 126.0f));
}

// Register allocation
constexpr int REG_LDST = 24;   // LdSt: minimal registers
constexpr int REG_MMA = 168;   // MMA: fragment-heavy
constexpr int REG_PREP = 160;  // Prep: scalar computation

// MMA atom alias
using MMA_TF32 = cute::SM80_16x8x8_F32TF32TF32F32_TN;

// Single-warp TiledMMA — each MMA warp handles its own 16-row block of the
// (64, 32) dQ/dQ2/dKt output. We branch on warp_id to select the active block.
using TiledMMA_BWD = decltype(make_tiled_mma(MMA_TF32{}));

// Helper: reinterpret float as uint32_t for MMA operand
__forceinline__ __device__ uint32_t
f2u(float f) {
    return reinterpret_cast<uint32_t&>(f);
}

// =====================================================================
// Thread Role Assignment
// =====================================================================
// threads   0-127: MMA
// threads 128-383: Prep
// threads 384-415: LdSt

enum class WGRole {
    LdSt = 0,
    MMA = 1,
    Prep = 2,
};

__forceinline__ __device__ WGRole
get_wg_role(int warp_idx) {
    if (warp_idx < NUM_MMA_WARPS)
        return WGRole::MMA;
    if (warp_idx < NUM_MMA_WARPS + NUM_PREP_WARPS)
        return WGRole::Prep;
    return WGRole::LdSt;
}

// =====================================================================
// SMEM Layouts
// =====================================================================

// Q, K: bf16 [64, 32] — TMA loaded
using SmemLayoutInputBF16 = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<T_TILE>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// G, dQ_in, dK_in, dG_in, dQ_out, dK_out: fp32 [64, 32] — TMA loaded / MMA R2S
using SmemLayoutInputFP32 = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<T_TILE>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dAqk, dAkk: fp32 [64, 64] — TMA loaded
using SmemLayoutDA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<float>{},
    Shape<Int<T_TILE>, Int<T_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KG, QG, KBG: fp32 [64, 32] — Prep element-wise, MMA thread load
template<int NUM_TILES>
using SmemLayoutMatBTF32Tranposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_32B_Atom<tf32>{},
    Shape<Int<K_TILE>, Int<SUB_T_TILE * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));


// =====================================================================
// SharedStorage (~201 KB, within 228 KB limit)
// =====================================================================
struct SharedStorage {
    // TMA-loaded input buffers (double-buffered per ki)
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> smem_q[NUM_BUF];   // 2 × 4 KB = 8 KB
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> smem_k[NUM_BUF];   // 2 × 4 KB = 8 KB
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_g[NUM_BUF];  // 2 × 8 KB = 16 KB

    // dAqk, dAkk: loaded once per tile (double-buffered across tiles)
    array_aligned<float, cosize_v<SmemLayoutDA>> smem_daqk[NUM_BUF_A];  // 16 KB
    array_aligned<float, cosize_v<SmemLayoutDA>> smem_dakk[NUM_BUF_A];  // 16 KB

    // dQ, dK, dG inter-chunk input (double-buffered per ki)
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_dq_in[NUM_BUF];  // 2 × 8 KB = 16 KB
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_dk_in[NUM_BUF];  // 2 × 8 KB = 16 KB
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_dg_in[NUM_BUF];  // 2 × 8 KB = 16 KB

    // KG
    struct {
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<1>>> intra[6];
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<1>>> inter[4];
    } kg_all;  // 20480 bytes, single-buffered
    
    // QG, KBG
    struct {
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<2>>> intra[6];
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<2>>> inter[4];
    } qkg_all; // 40960 bytes, single-buffered

    // Scalar data
    array_aligned<float, T_TILE> beta_smem[NUM_BUF_A];  // single-buffered with dA
    array_aligned<float, T_TILE> db_accum;   // 256 B (cross-ki dB accumulator)

    // ── MMA-result scratch: dQ, dQ2 (also reused as DKT exchange pair). dKt
    //    reads share smem_mma_dq (zero placeholder). Two buffers suffice for
    //    DKT exchange (sDKT_0 / sDKT_1).
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dq;    // dQ (intra) + DKT_0
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dq2;   // dQ2 (inter) + DKT_1
    // Upper-half read path: dk_lower = dAkk @ KG (analogue of dq but with dAkk).
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dk_lower_intra;
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dk_lower_inter;

    // ── Per-iteration db reduce scratch (WG0 -> WG1)
    array_aligned<float, T_TILE> db_partial;  // 256 B

    // ── Pipeline barriers ──
    // Load → consumers
    alignas(16) cute::uint64_t bar_load_qkg_ready[NUM_BUF];
    alignas(16) cute::uint64_t bar_load_dA_ready[NUM_BUF_A];

    // Prep → MMA
    alignas(16) cute::uint64_t bar_kg_all_ready, bar_qkg_all_ready;

    // Async pipelines: MMA → Prep (phase-tracked, single-buffered)
    alignas(16) cute::uint64_t bar_mma_dq_done;   // MMA signals: dQ + dQ2 ready
    alignas(16) cute::uint64_t bar_mma_dkt_done;   // MMA signals: dKt ready
    // Prep → MMA: dQ/dQ2 consumed, safe to overwrite smem_mma_dq for dKt.
    alignas(16) cute::uint64_t bar_prep_dq_consumed;

    // Prep → Load
    alignas(16) cute::uint64_t bar_dA_free[NUM_BUF_A];
    alignas(16) cute::uint64_t bar_buf_free[NUM_BUF];  // value-buffer free signal
};

// =====================================================================
// TMA Parameters Structure
// =====================================================================
template <
    typename ShapeQKG,
    typename ShapeDA,
    typename TMA_Q,
    typename TMA_K,
    typename TMA_G,
    typename TMA_DAqk,
    typename TMA_DAkk,
    typename TMA_DQ,
    typename TMA_DK,
    typename TMA_DG>
struct TmaParams {
    ShapeQKG shape_qkg;
    ShapeDA shape_da;
    TMA_Q tma_q;
    TMA_K tma_k;
    TMA_G tma_g;
    TMA_DAqk tma_dAqk;
    TMA_DAkk tma_dAkk;
    TMA_DQ tma_dq;
    TMA_DK tma_dk;
    TMA_DG tma_dg;
};

using TileScheduler = StaticPersistentTileScheduler;

// =====================================================================
// Prep + Epilogue body — ported from SM100 ComputeEpilogue
// =====================================================================
// Per-tile: K_ITERATION=4 ki passes. Each WG (0/1) handles K_OFF = WG_IDX*16
// of the K_TILE=32 split. Uses:
//   - bar_load_dA_ready[buf_idx_A]  : dA + beta loaded
//   - bar_load_qkg_ready[buf_idx_value] : Q/K/G/dQ/dK/dG loaded
//   - bar_kg_all_ready / bar_qkg_all_ready : Prep -> MMA (256-thread arrival)
//   - bar_buf_free[buf_idx_value], bar_dA_free[buf_idx_A] : Prep -> Load
//
// MMA-side dQ/dQ2/dKt are read from `smem_mma_dq[2]/dkt` (zero placeholders
// until MMA is implemented; thus the intra-MMA contributions evaluate to 0).
template <int WG_IDX, typename ParamsT>
__forceinline__ __device__ void
prep_compute_epilogue_body(
    SharedStorage* smem, const ParamsT& params,
    int idx_in_warpgroup,
    int& state_phase, int& buf_idx_A, int& buf_idx_value,
    int batch_idx, int head_idx, int tile_idx,
    int start_offset, int sub_seq_len,
    int A_phase_in) {

    constexpr int HALF_K = K_TILE / 2;
    constexpr int K_OFF  = WG_IDX * HALF_K;
    constexpr int DKT_BAR_ID = 4 + WG_IDX * 2;  // 4 / 6 (avoid 0/1/2 collisions)
    constexpr int DB_REDUCE_BAR_ID = 1;

    int local_idx = idx_in_warpgroup % 64;

    // Init db: WG1 upper-half loads existing db_in
    float db = 0.0f;
    if constexpr (WG_IDX == 1) {
        if (idx_in_warpgroup >= 64 && local_idx < sub_seq_len) {
            db = reinterpret_cast<float*>(params.db_ptr)
                [(start_offset + tile_idx * T_TILE + local_idx) * params.h + head_idx];
        }
    }

    // b_phase tracks the phase parity for MMA<->Prep async bars (flips per k_idx).
    int b_phase = 0;

    for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
        int local_phase = (state_phase >> buf_idx_value) & 1;

        // Wait for Q/K/G/dQ/dK/dG.  Ready-bar wait(local_phase): first iter
        // wait(0) on fresh bar (parity 0) blocks until producer (LdSt TMA)
        // arrives, then returns.
        cute::wait_barrier(smem->bar_load_qkg_ready[buf_idx_value], local_phase);

        Tensor sQ = make_tensor(make_smem_ptr(smem->smem_q[buf_idx_value].data()), SmemLayoutInputBF16{});
        Tensor sK = make_tensor(make_smem_ptr(smem->smem_k[buf_idx_value].data()), SmemLayoutInputBF16{});
        Tensor sG = make_tensor(make_smem_ptr(smem->smem_g[buf_idx_value].data()), SmemLayoutInputFP32{});

        int y = idx_in_warpgroup % 8 * 4;
        constexpr int kg_offset  = SUB_T_TILE * K_TILE;
        constexpr int qkg_offset = SUB_T_TILE * K_TILE * 2;

        Tensor sKG_intra  = make_tensor(make_smem_ptr(smem->kg_all.intra[0].data()),  SmemLayoutMatBTF32Tranposed<1>{});
        Tensor sQKG_intra = make_tensor(make_smem_ptr(smem->qkg_all.intra[0].data()), SmemLayoutMatBTF32Tranposed<2>{});

        // ── kg_intra (non-overlapping rows) ──
        if constexpr (WG_IDX == 0) {
            float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
            setup_kg_intra<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(
                sG, sK, sKG_intra, 0, idx_in_warpgroup, gn3, 3);
        } else {
            float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
            float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
            setup_kg_intra_2gn<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(
                sG, sK, sKG_intra, 0, idx_in_warpgroup, gn1, gn2, 0, 1);
            setup_kg_intra<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(
                sG, sK, sKG_intra, 1, idx_in_warpgroup, gn2, 2);
        }

        // ── fused intra (kg + qkg shared rows) ──
        {
            float2 beta_v[4];
            if constexpr (WG_IDX == 0) {
                float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
                float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
                for (int j = 1; j <= 2; ++j) {
                    int x = idx_in_warpgroup / 8 + j * 16;
                    if (x < sub_seq_len) {
                        beta_v[j] = __bfloat1622float2(__bfloat162bfloat162(
                            (__nv_bfloat16)smem->beta_smem[buf_idx_A][x]));
                    }
                }
                setup_intra_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_intra), decltype(sQKG_intra), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_intra, sQKG_intra, 1, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta_v[1], 4, 0);
                setup_intra_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_intra), decltype(sQKG_intra), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_intra, sQKG_intra, 2, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta_v[2], 5, 1);
            }
            // WG_IDX == 1: no fused intra rows (handled by qkg_intra below)
        }

        // ── fused inter (kg + qkg) ──
        {
            Tensor sKG_inter  = make_tensor(make_smem_ptr(smem->kg_all.inter[0].data()),  SmemLayoutMatBTF32Tranposed<1>{});
            Tensor sQKG_inter = make_tensor(make_smem_ptr(smem->qkg_all.inter[0].data()), SmemLayoutMatBTF32Tranposed<2>{});
            float2 beta_v[4];
            if constexpr (WG_IDX == 0) {
                beta_v[0] = __bfloat1622float2(__bfloat162bfloat162(
                    (__nv_bfloat16)smem->beta_smem[buf_idx_A][idx_in_warpgroup / 8]));
                int x3 = idx_in_warpgroup / 8 + 48;
                if (x3 < sub_seq_len) {
                    beta_v[3] = __bfloat1622float2(__bfloat162bfloat162(
                        (__nv_bfloat16)smem->beta_smem[buf_idx_A][x3]));
                }
                float4 gn_h0, gn_h3;
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 0, idx_in_warpgroup, sub_seq_len, beta_v[0], gn_h0);
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 3, idx_in_warpgroup, sub_seq_len, beta_v[3], gn_h3);
            } else {
                int x1 = idx_in_warpgroup / 8 + 16;
                if (x1 < sub_seq_len) {
                    beta_v[1] = __bfloat1622float2(__bfloat162bfloat162(
                        (__nv_bfloat16)smem->beta_smem[buf_idx_A][x1]));
                }
                int x2 = idx_in_warpgroup / 8 + 32;
                if (x2 < sub_seq_len) {
                    beta_v[2] = __bfloat1622float2(__bfloat162bfloat162(
                        (__nv_bfloat16)smem->beta_smem[buf_idx_A][x2]));
                }
                float4 gn_h1, gn_h2;
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 1, idx_in_warpgroup, sub_seq_len, beta_v[1], gn_h1);
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 2, idx_in_warpgroup, sub_seq_len, beta_v[2], gn_h2);
            }
        }

        fence_view_async_shared();
        cute::arrive_barrier(smem->bar_kg_all_ready);

        // ── intra scale (overlap with MMA kg phase) ──
        float scale[HALF_K];
        epilogue_compute_intra_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, scale);

        // ── qkg_intra (non-overlapping rows) ──
        {
            float2 beta_v[4];
            if constexpr (WG_IDX == 0) {
                float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
                int x3 = idx_in_warpgroup / 8 + 48;
                if (x3 < sub_seq_len) {
                    beta_v[3] = __bfloat1622float2(__bfloat162bfloat162(
                        (__nv_bfloat16)smem->beta_smem[buf_idx_A][x3]));
                }
                setup_qkg_intra<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(
                    sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta_v[3], gn1, 2);
            } else {
                float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
                float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
                int x2 = idx_in_warpgroup / 8 + 32;
                if (x2 < sub_seq_len) {
                    beta_v[2] = __bfloat1622float2(__bfloat162bfloat162(
                        (__nv_bfloat16)smem->beta_smem[buf_idx_A][x2]));
                }
                int x3 = idx_in_warpgroup / 8 + 48;
                if (x3 < sub_seq_len) {
                    beta_v[3] = __bfloat1622float2(__bfloat162bfloat162(
                        (__nv_bfloat16)smem->beta_smem[buf_idx_A][x3]));
                }
                setup_qkg_intra<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(
                    sG, sQ, sK, sQKG_intra, 2, idx_in_warpgroup, sub_seq_len, beta_v[2], gn2, 3);
                setup_qkg_intra_2gn<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(
                    sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta_v[3], gn2, gn3, 4, 5);
            }
        }

        fence_view_async_shared();
        cute::arrive_barrier(smem->bar_qkg_all_ready);

        // ── DEBUG: print B-operand checksums (one tile, one warp per WG) ──
#if KDA_BWD_SM90_DEBUG_PRINT
        if (blockIdx.x == 0 && tile_idx == 0 && batch_idx == 0 && head_idx == 0 &&
            k_idx == 0 && idx_in_warpgroup == 0) {
            float kg_sum = 0.f, qkg_sum = 0.f;
            float* p_kg  = reinterpret_cast<float*>(smem->kg_all.intra[0].data());
            float* p_qkg = reinterpret_cast<float*>(smem->qkg_all.intra[0].data());
            for (int i = 0; i < 16; ++i) { kg_sum += p_kg[i]; qkg_sum += p_qkg[i]; }
            printf("[PREP WG%d] tile=%d k_idx=%d sub_seq_len=%d kg.intra[0..16]=%.4f qkg.intra[0..16]=%.4f\n",
                   WG_IDX, tile_idx, k_idx, sub_seq_len, kg_sum, qkg_sum);
        }
#endif

        // ── Wait for MMA dQ+dQ2 (+dk_lower) ready, then read and apply intra scale ──
        cute::wait_barrier(smem->bar_mma_dq_done, b_phase);
        Tensor sMmaDQ_lo = make_tensor(make_smem_ptr(smem->smem_mma_dq.data()),  SmemLayoutInputFP32{});
        Tensor sMmaDQ_up = make_tensor(make_smem_ptr(smem->smem_mma_dk_lower_intra.data()), SmemLayoutInputFP32{});
        float res[HALF_K];
        if (idx_in_warpgroup < 64) {
            epilogue_apply_dq_intra_smem<HALF_K, K_OFF>(sMmaDQ_lo, idx_in_warpgroup, res, scale);
        } else {
            epilogue_apply_dq_intra_smem<HALF_K, K_OFF>(sMmaDQ_up, idx_in_warpgroup, res, scale);
        }

        // ── Compute inter scale directly from sG ──
        {
            int row = idx_in_warpgroup % 64;
            int g_half_row = min(row / 16 * 16 + 8, sub_seq_len - 1);
            for (int i = 0; i < HALF_K / 4; ++i) {
                float4 bg      = *reinterpret_cast<float4*>(&sG(row,        K_OFF + i * 4));
                float4 bg_half = *reinterpret_cast<float4*>(&sG(g_half_row, K_OFF + i * 4));
                float2 d0 = float2_sub(reinterpret_cast<float2*>(&bg)[0], reinterpret_cast<float2*>(&bg_half)[0]);
                float2 d1 = float2_sub(reinterpret_cast<float2*>(&bg)[1], reinterpret_cast<float2*>(&bg_half)[1]);
                scale[i * 4]     = exp2f(d0.x);
                scale[i * 4 + 1] = exp2f(d0.y);
                scale[i * 4 + 2] = exp2f(d1.x);
                scale[i * 4 + 3] = exp2f(d1.y);
            }
        }

        // ── Combine dq2 / dk_lower_inter ──
        Tensor sMmaDQ2_lo = make_tensor(make_smem_ptr(smem->smem_mma_dq2.data()),           SmemLayoutInputFP32{});
        Tensor sMmaDQ2_up = make_tensor(make_smem_ptr(smem->smem_mma_dk_lower_inter.data()), SmemLayoutInputFP32{});
        if (idx_in_warpgroup < 64) {
            epilogue_combine_dq_inter_smem<HALF_K, K_OFF>(sMmaDQ2_lo, idx_in_warpgroup, res, scale);
        } else {
            epilogue_combine_dq_inter_smem<HALF_K, K_OFF>(sMmaDQ2_up, idx_in_warpgroup, res, scale);
        }

        // ── Output dq / accumulate db ──
        {
            if (idx_in_warpgroup >= 64) {
                bf16 beta_val = (local_idx < sub_seq_len)
                    ? (bf16)(__nv_bfloat16)smem->beta_smem[buf_idx_A][local_idx]
                    : bf16{};
                float* db_out_addr = nullptr;
                epilogue_accumulate_db<HALF_K, K_OFF>(
                    sK, idx_in_warpgroup, sub_seq_len, res, db, false, db_out_addr, beta_val);
            } else {
                Tensor sDQ_in = make_tensor(make_smem_ptr(smem->smem_dq_in[buf_idx_value].data()), SmemLayoutInputFP32{});
                float* dq_out_base = (float*)(params.dq_out_ptr)
                    + (start_offset + tile_idx * T_TILE + local_idx) * params.h * K_SIZE
                    + head_idx * K_SIZE + k_idx * K_TILE + K_OFF;
                epilogue_output_dq<HALF_K, K_OFF>(sQ, sDQ_in, idx_in_warpgroup, sub_seq_len, res, dq_out_base);
            }
        }

        // ── Per-iter db reduce: WG0 partial -> smem -> WG1 accumulates ──
        if (idx_in_warpgroup >= 64) {
            if constexpr (WG_IDX == 0) {
                if (local_idx < sub_seq_len) {
                    smem->db_partial[local_idx] = db;
                }
            }
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, DB_REDUCE_BAR_ID);
            if constexpr (WG_IDX == 1) {
                if (local_idx < sub_seq_len) {
                    db += smem->db_partial[local_idx];
                }
            }
            if constexpr (WG_IDX == 0) {
                db = 0.0f;
            }
        }

        // ── dkt scale (from sG) ──
        epilogue_compute_dkt_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, sub_seq_len, scale);

        // ── Read MMA dKt from SMEM ──
        // Intra off-diag dkt lives in smem_mma_dq (lower-half scale exp2(g[next_top]-g[row])).
        // Inter diag dkt lives in smem_mma_dq2 (upper-half scale exp2(g[mid]-g[row])).
        cute::arrive_barrier(smem->bar_prep_dq_consumed);
        cute::wait_barrier(smem->bar_mma_dkt_done, b_phase);
        Tensor sMmaDKt_intra = make_tensor(make_smem_ptr(smem->smem_mma_dq.data()),  SmemLayoutInputFP32{});
        Tensor sMmaDKt_inter = make_tensor(make_smem_ptr(smem->smem_mma_dq2.data()), SmemLayoutInputFP32{});
        float res_dkt[HALF_K];
        if (idx_in_warpgroup < 64) {
            epilogue_process_dkt_smem<HALF_K, K_OFF>(sMmaDKt_intra, idx_in_warpgroup, res_dkt, scale, sub_seq_len);
        } else {
            epilogue_process_dkt_smem<HALF_K, K_OFF>(sMmaDKt_inter, idx_in_warpgroup, res_dkt, scale, sub_seq_len);
        }

        // ── DKT exchange between lower/upper within WG (128-thread named barrier) ──
        NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);
        Tensor sDKT_0 = make_tensor(make_smem_ptr(smem->smem_mma_dq.data()), SmemLayoutInputFP32{});
        Tensor sDKT_1 = make_tensor(make_smem_ptr(smem->smem_mma_dq2.data()), SmemLayoutInputFP32{});
        epilogue_exchange_dkt<HALF_K, K_OFF>(sDKT_0, sDKT_1, idx_in_warpgroup, res, res_dkt);
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);

        // ── Output dg / dk ──
        if (idx_in_warpgroup < 64) {
            if (local_idx < sub_seq_len) {
                Tensor sDG_in = make_tensor(make_smem_ptr(smem->smem_dg_in[buf_idx_value].data()), SmemLayoutInputFP32{});
                float* dg_out_base = (float*)(params.dg_out_ptr)
                    + (start_offset + tile_idx * T_TILE + local_idx) * params.h * K_SIZE
                    + head_idx * K_SIZE + k_idx * K_TILE + K_OFF;
                epilogue_output_dg<HALF_K, K_OFF>(sK, sDG_in, sDKT_1, idx_in_warpgroup, res, res_dkt, dg_out_base);
            }
        } else {
            Tensor sDK_in = make_tensor(make_smem_ptr(smem->smem_dk_in[buf_idx_value].data()), SmemLayoutInputFP32{});
            float* dk_out_base = (float*)(params.dk_out_ptr)
                + (start_offset + tile_idx * T_TILE + local_idx) * params.h * K_SIZE
                + head_idx * K_SIZE + k_idx * K_TILE + K_OFF;
            epilogue_output_dk<HALF_K, K_OFF>(sDK_in, sDKT_0, idx_in_warpgroup, sub_seq_len, res, res_dkt, dk_out_base);
        }

        // Release Q/K/G/dQ/dK/dG buffer slot for the next tile's Load.
        cute::arrive_barrier(smem->bar_buf_free[buf_idx_value]);
        state_phase ^= 1 << buf_idx_value;
        buf_idx_value = (buf_idx_value + 1) % NUM_BUF;
        b_phase ^= 1;

        // Lock-step WG0 and WG1 at each k_idx so DB_REDUCE_BAR pairings stay
        // matched across iterations.
        NamedBarrier::arrive_and_wait(256, /*PREP_ITER_BAR_ID*/ 7);
    }

    // ── Final db output: WG1 upper half holds the fully-reduced db ──
    if constexpr (WG_IDX == 1) {
        if (idx_in_warpgroup >= 64 && local_idx < sub_seq_len) {
            reinterpret_cast<float*>(params.db_out_ptr)
                [(start_offset + tile_idx * T_TILE + local_idx) * params.h + head_idx] = db;
#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0 && tile_idx == 0 && batch_idx == 0 && head_idx == 0 && local_idx == 0) {
                printf("[PREP WG1] tile=%d wrote db_out[0]=%.6f\n", tile_idx, db);
            }
#endif
        }
    }
}

// =====================================================================
// Kernel Entry Point — Phase 1 Skeleton
// =====================================================================
template <typename TmaParamsT>
__global__ void
__launch_bounds__(NUM_THREADS, 1) kda_bwd_intra_sm90_kernel(
    __grid_constant__ const KDA_bwd_intra_params params, __grid_constant__ const TmaParamsT tma_params) {
    const int thread_idx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warp = thread_idx % 32;
    WGRole role = get_wg_role(warp_idx);

#if KDA_BWD_SM90_DEBUG_PRINT
    if (blockIdx.x == 0 && thread_idx == 0) {
        printf("[KERNEL] enter blockIdx.x=0 NUM_THREADS=%d\n", NUM_THREADS);
    }
#endif

    TileScheduler tile_scheduler{params.tile_scheduler_params};

    extern __shared__ char shared_buf[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(shared_buf);

    int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
    int* cu_len_ptr = (int*)params.cu_seqlens_ptr;

    // ── Barrier initialization (Load warp / elected thread) ──
    if (warp_idx == LOAD_WARP_IDX && elect_one_sync()) {
        // Prefetch TMA descriptors
        cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dAqk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dAkk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dq.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dg.get_tma_descriptor());

        // Load → consumers
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_load_qkg_ready[i], 1);
        }
        for (int i = 0; i < NUM_BUF_A; ++i) {
            cute::initialize_barrier(smem->bar_load_dA_ready[i], 2);  // TMA + elected Load thread
        }

        // Prep → MMA
        cute::initialize_barrier(smem->bar_kg_all_ready, 256);
        cute::initialize_barrier(smem->bar_qkg_all_ready, 256);

        // Async barriers: MMA → Prep / Prep → MMA
        cute::initialize_barrier(smem->bar_mma_dq_done, 128);    // MMA WG arrival
        cute::initialize_barrier(smem->bar_mma_dkt_done, 128);   // MMA WG arrival
        cute::initialize_barrier(smem->bar_prep_dq_consumed, 256); // Both Prep WGs

        // Prep → Load.  No pre-arrive needed.  Phase semantics:
        // `wait_barrier(bar, P)` blocks while current phase parity == P
        // (returns when they differ).  Free-bar consumers (LdSt) use
        // wait(local_phase ^ 1): first iter wait(1) on fresh bar (parity 0)
        // returns immediately, signalling buffer is initially free.
        for (int i = 0; i < NUM_BUF_A; ++i) {
            cute::initialize_barrier(smem->bar_dA_free[i], 256);
        }
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_buf_free[i], 256);
        }
        cutlass::arch::fence_barrier_init();
    }

    __syncthreads();

    // ── Zero-init db_partial scratch (MMA writes smem_mma_dq/dq2 directly). ──
    if (thread_idx < T_TILE) smem->db_partial[thread_idx] = 0.0f;
    __syncthreads();
    // state_phase packs the current phase bit of each double-buffered resource:
    // low NUM_BUF bits track value-buffer phases, high NUM_BUF bits track dA/beta phases.
    int state_phase = 0;
    // buf_idx_A selects which dA/beta slot is currently owned by this tile.
    int buf_idx_A = 0;
    // buf_idx_value selects which per-ki Q/K/G/dQ/dK/dG slot is active in the current step.
    int buf_idx_value = 0;

    // =================================================================
    // Load warp — TMA loads + writes
    // =================================================================
    if (role == WGRole::LdSt) {
        if (elect_one_sync()) {
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int A_phase = (state_phase >> (buf_idx_A + NUM_BUF)) & 1;
#if KDA_BWD_SM90_DEBUG_PRINT
                if (blockIdx.x == 0) printf("[LDST] before wait bar_dA_free buf=%d expect_phase=%d\n", buf_idx_A, A_phase);
#endif
                cute::wait_barrier(smem->bar_dA_free[buf_idx_A], A_phase ^ 1);
#if KDA_BWD_SM90_DEBUG_PRINT
                if (blockIdx.x == 0) printf("[LDST] after wait bar_dA_free buf=%d\n", buf_idx_A);
#endif

                int tid = tile_scheduler.get_current_tile_id();

                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);
                int token_offset = cu_len_ptr[batch_idx];
                int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
                int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);


                // TMA load dAqk, dAkk.
                {
                    Tensor sDAqk = make_tensor(make_smem_ptr(smem->smem_daqk[buf_idx_A].data()), SmemLayoutDA{});
                    Tensor sDAkk = make_tensor(make_smem_ptr(smem->smem_dakk[buf_idx_A].data()), SmemLayoutDA{});
                    int tma_bytes_da = sizeof(make_tensor_like(sDAqk)) + sizeof(make_tensor_like(sDAkk));

                    Tensor mDaqk = domain_offset(
                        make_coord(token_offset, _0{}, _0{}), tma_params.tma_dAqk.get_tma_tensor(tma_params.shape_da));
                    Tensor mDakk = domain_offset(
                        make_coord(token_offset, _0{}, _0{}), tma_params.tma_dAkk.get_tma_tensor(tma_params.shape_da));
                    Tensor gDaqk = local_tile(
                        mDaqk(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<T_TILE>{}), make_coord(tile_idx, _0{}));
                    Tensor gDakk = local_tile(
                        mDakk(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<T_TILE>{}), make_coord(tile_idx, _0{}));

                    cute::set_barrier_transaction_bytes(smem->bar_load_dA_ready[buf_idx_A], tma_bytes_da);
                    launch_tma_copy(tma_params.tma_dAqk, gDaqk, sDAqk, smem->bar_load_dA_ready[buf_idx_A]);
                    launch_tma_copy(tma_params.tma_dAkk, gDakk, sDAkk, smem->bar_load_dA_ready[buf_idx_A]);
                }

                // Load beta via the same elected Load thread.
                {
                    float* beta_base = (float*)params.beta_ptr;
                    for (int i = 0; i < T_TILE; ++i) {
                        smem->beta_smem[buf_idx_A][i] =
                            (i < sub_seq_len) ? beta_base[(token_offset + tile_idx * T_TILE + i) * params.h + head_idx]
                                              : 0.0f;
                    }
                    fence_view_async_shared();
                }

                // Signal Prep/MMA: dA + beta ready.
                cute::arrive_barrier(smem->bar_load_dA_ready[buf_idx_A]);

                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    int local_phase = (state_phase >> buf_idx_value) & 1;
                    cute::wait_barrier(smem->bar_buf_free[buf_idx_value], local_phase ^ 1);

                    // TMA load Q, K, G + dQ, dK, dG inter-chunk.
                    {
                        Tensor sQ = make_tensor(make_smem_ptr(smem->smem_q[buf_idx_value].data()), SmemLayoutInputBF16{});
                        Tensor sK = make_tensor(make_smem_ptr(smem->smem_k[buf_idx_value].data()), SmemLayoutInputBF16{});
                        Tensor sG = make_tensor(make_smem_ptr(smem->smem_g[buf_idx_value].data()), SmemLayoutInputFP32{});
                        Tensor sDQ =
                            make_tensor(make_smem_ptr(smem->smem_dq_in[buf_idx_value].data()), SmemLayoutInputFP32{});
                        Tensor sDK =
                            make_tensor(make_smem_ptr(smem->smem_dk_in[buf_idx_value].data()), SmemLayoutInputFP32{});
                        Tensor sDG =
                            make_tensor(make_smem_ptr(smem->smem_dg_in[buf_idx_value].data()), SmemLayoutInputFP32{});

                        int tma_bytes = sizeof(make_tensor_like(sQ)) + sizeof(make_tensor_like(sK)) +
                                        sizeof(make_tensor_like(sG)) + sizeof(make_tensor_like(sDQ)) +
                                        sizeof(make_tensor_like(sDK)) + sizeof(make_tensor_like(sDG));

                        Tensor mQ = domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_q.get_tma_tensor(tma_params.shape_qkg));
                        Tensor mK = domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_k.get_tma_tensor(tma_params.shape_qkg));
                        Tensor mG = domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_g.get_tma_tensor(tma_params.shape_qkg));
                        Tensor mDQ = domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_dq.get_tma_tensor(tma_params.shape_qkg));
                        Tensor mDK = domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_dk.get_tma_tensor(tma_params.shape_qkg));
                        Tensor mDG = domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_dg.get_tma_tensor(tma_params.shape_qkg));

                        Tensor gQ = local_tile(
                            mQ(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                        Tensor gK = local_tile(
                            mK(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                        Tensor gG = local_tile(
                            mG(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                        Tensor gDQ = local_tile(
                            mDQ(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                        Tensor gDK = local_tile(
                            mDK(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                        Tensor gDG = local_tile(
                            mDG(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));

                        cute::set_barrier_transaction_bytes(smem->bar_load_qkg_ready[buf_idx_value], tma_bytes);
                        launch_tma_copy(tma_params.tma_q, gQ, sQ, smem->bar_load_qkg_ready[buf_idx_value]);
                        launch_tma_copy(tma_params.tma_k, gK, sK, smem->bar_load_qkg_ready[buf_idx_value]);
                        launch_tma_copy(tma_params.tma_g, gG, sG, smem->bar_load_qkg_ready[buf_idx_value]);
                        launch_tma_copy(tma_params.tma_dq, gDQ, sDQ, smem->bar_load_qkg_ready[buf_idx_value]);
                        launch_tma_copy(tma_params.tma_dk, gDK, sDK, smem->bar_load_qkg_ready[buf_idx_value]);
                        launch_tma_copy(tma_params.tma_dg, gDG, sDG, smem->bar_load_qkg_ready[buf_idx_value]);
                    }

                    state_phase ^= 1 << buf_idx_value;
                    buf_idx_value = (buf_idx_value + 1) % NUM_BUF;
                }

                state_phase ^= 1 << (buf_idx_A + NUM_BUF);
                buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
#if KDA_BWD_SM90_DEBUG_PRINT
                if (blockIdx.x == 0) printf("[LDST] tile complete\n");
#endif
            }
#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0) printf("[LDST] outer loop exit\n");
#endif
        }

        // =================================================================
        // MMA warpgroup — mma.sync 4-pass sub-chunk loop
        // =================================================================
    } else if (role == WGRole::MMA) {
        // cutlass::arch::warpgroup_reg_alloc<REG_MMA>();  // optional; enable once stable

        const int warp_id_in_wg = thread_idx / 32;   // 0..3 — selects query-block M=[16*w, 16*(w+1))
        const int lane_id = thread_idx % 32;

        // Single-warp TiledMMA (atom 16x8x8). Each MMA warp independently computes
        // a 16-row slice of the (T_TILE=64, K_TILE=32) dQ/dQ2/dKt tile, then R2S
        // into its own quarter of smem_mma_dq / smem_mma_dq2.
        TiledMMA_BWD tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(lane_id);

        // ── Slot → (active warp, dAqk column-block) mapping (see docs/sm90_bwd_mma_plan.md) ──
        // kg_intra slot s contributes to dQ for query block kg_intra_warp[s], using
        // dAqk columns kg_intra_k[s]*16 .. +16 and KG pre-gated against gn_{q_block}.
        constexpr int kg_intra_warp[6] = {1, 2, 2, 3, 3, 3};
        constexpr int kg_intra_k[6]    = {0, 0, 1, 0, 1, 2};
        // qkg_intra slot layout matches Prep:
        //   slot 0: (i=1, j=0)   slot 1: (i=2, j=0)
        //   slot 2: (i=3, j=0)   slot 3: (i=2, j=1)
        //   slot 4: (i=3, j=1)   slot 5: (i=3, j=2)
        constexpr int qkg_intra_j[6] = {0, 0, 0, 1, 1, 2};
        constexpr int qkg_intra_i[6] = {1, 2, 3, 2, 3, 3};

        // Type aliases for staged fragments
        using FragC_t   = decltype(partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{}));

        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int A_phase = (state_phase >> (buf_idx_A + NUM_BUF)) & 1;
            cute::wait_barrier(smem->bar_load_dA_ready[buf_idx_A], A_phase);

            int tid = tile_scheduler.get_current_tile_id();
            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx  = get<1>(blk_coord);
            int tile_idx  = get<2>(blk_coord);
            int seq_len   = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0 && thread_idx == 0 && tile_idx == 0 && batch_idx == 0 && head_idx == 0) {
                printf("[MMA] tile=%d sub_seq_len=%d begin\n", tile_idx, sub_seq_len);
            }
#endif

            Tensor sDAqk = make_tensor(make_smem_ptr(smem->smem_daqk[buf_idx_A].data()), SmemLayoutDA{});
            Tensor sDAkk = make_tensor(make_smem_ptr(smem->smem_dakk[buf_idx_A].data()), SmemLayoutDA{});

            int b_phase = 0;

            for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                // === KG PHASE: dQ (intra off-diag) + dQ2 (inter diag) ===
                cute::wait_barrier(smem->bar_kg_all_ready, b_phase);

                FragC_t tDQ_acc   = partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{});
                FragC_t tDQ2_acc  = partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{});
                FragC_t tDKL_acc  = partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{});
                FragC_t tDKL2_acc = partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{});
                clear(tDQ_acc);
                clear(tDQ2_acc);
                clear(tDKL_acc);
                clear(tDKL2_acc);

                // ---- KG intra: 6 strict-off-diagonal (q_block, k_block) pairs ----
                CUTE_UNROLL
                for (int s = 0; s < 6; ++s) {
                    if (warp_id_in_wg != kg_intra_warp[s]) continue;
                    int q_block = kg_intra_warp[s];
                    int k_block = kg_intra_k[s];

                    auto sA_qk = local_tile(sDAqk, Shape<_16, _16>{}, make_coord(q_block, k_block));
                    auto sA_kk = local_tile(sDAkk, Shape<_16, _16>{}, make_coord(q_block, k_block));
                    auto tAsA_qk = thr_mma.partition_A(sA_qk);
                    auto tAsA_kk = thr_mma.partition_A(sA_kk);
                    auto tArA_qk = make_fragment_like(tAsA_qk);
                    auto tArA_kk = make_fragment_like(tAsA_kk);
                    cute::copy(tAsA_qk, tArA_qk);
                    cute::copy(tAsA_kk, tArA_kk);

                    // sub_seq_len-bounds guard (strict off-diag — no causal mask needed).
                    {
                        auto cA = make_identity_tensor(Shape<_16, _16>{});
                        auto tAcA = thr_mma.partition_A(cA);
                        CUTE_UNROLL
                        for (int i = 0; i < size(tArA_qk); ++i) {
                            int r = get<0>(tAcA(i));
                            int c = get<1>(tAcA(i));
                            int g_r = q_block * 16 + r;
                            int g_c = k_block * 16 + c;
                            if (g_r >= sub_seq_len || g_c >= sub_seq_len) {
                                tArA_qk(i) = 0.0f;
                                tArA_kk(i) = 0.0f;
                            }
                        }
                    }

                    Tensor sB = make_tensor(make_smem_ptr(smem->kg_all.intra[s].data()),
                                            SmemLayoutMatBTF32Tranposed<1>{});
                    auto tBsB = thr_mma.partition_B(sB);
                    auto tBrB = thr_mma.partition_fragment_B(sB);
                    cute::copy(tBsB, tBrB);

                    auto tArA_qk_tf32 = recast<tfloat32_t>(tArA_qk);
                    auto tArA_kk_tf32 = recast<tfloat32_t>(tArA_kk);
                    auto tBrB_tf32    = recast<tfloat32_t>(tBrB);
                    cute::gemm(tiled_mma, tArA_qk_tf32, tBrB_tf32, tDQ_acc);
                    cute::gemm(tiled_mma, tArA_kk_tf32, tBrB_tf32, tDKL_acc);
                }

                // ---- KG inter: 4 diagonal blocks with causal mask ----
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s) {
                    if (warp_id_in_wg != s) continue;
                    int i_block = s;

                    auto sA_qk = local_tile(sDAqk, Shape<_16, _16>{}, make_coord(i_block, i_block));
                    auto sA_kk = local_tile(sDAkk, Shape<_16, _16>{}, make_coord(i_block, i_block));
                    auto tAsA_qk = thr_mma.partition_A(sA_qk);
                    auto tAsA_kk = thr_mma.partition_A(sA_kk);
                    auto tArA_qk = make_fragment_like(tAsA_qk);
                    auto tArA_kk = make_fragment_like(tAsA_kk);
                    cute::copy(tAsA_qk, tArA_qk);
                    cute::copy(tAsA_kk, tArA_kk);

                    {
                        auto cA = make_identity_tensor(Shape<_16, _16>{});
                        auto tAcA = thr_mma.partition_A(cA);
                        CUTE_UNROLL
                        for (int i = 0; i < size(tArA_qk); ++i) {
                            int r = get<0>(tAcA(i));
                            int c = get<1>(tAcA(i));
                            int g_r = i_block * 16 + r;
                            int g_c = i_block * 16 + c;
                            bool keep = (r >= c) && (g_r < sub_seq_len) && (g_c < sub_seq_len);
                            if (!keep) {
                                tArA_qk(i) = 0.0f;
                                tArA_kk(i) = 0.0f;
                            }
                        }
                    }

                    Tensor sB = make_tensor(make_smem_ptr(smem->kg_all.inter[s].data()),
                                            SmemLayoutMatBTF32Tranposed<1>{});
                    auto tBsB = thr_mma.partition_B(sB);
                    auto tBrB = thr_mma.partition_fragment_B(sB);
                    cute::copy(tBsB, tBrB);

                    auto tArA_qk_tf32 = recast<tfloat32_t>(tArA_qk);
                    auto tArA_kk_tf32 = recast<tfloat32_t>(tArA_kk);
                    auto tBrB_tf32    = recast<tfloat32_t>(tBrB);
                    cute::gemm(tiled_mma, tArA_qk_tf32, tBrB_tf32, tDQ2_acc);
                    cute::gemm(tiled_mma, tArA_kk_tf32, tBrB_tf32, tDKL2_acc);
                }

                // R2S: each warp writes its 16-row slice to smem_mma_dq / smem_mma_dq2 / dk_lower buffers.
                {
                    Tensor sDQ   = make_tensor(make_smem_ptr(smem->smem_mma_dq.data()),               SmemLayoutInputFP32{});
                    Tensor sDQ2  = make_tensor(make_smem_ptr(smem->smem_mma_dq2.data()),              SmemLayoutInputFP32{});
                    Tensor sDKL  = make_tensor(make_smem_ptr(smem->smem_mma_dk_lower_intra.data()),   SmemLayoutInputFP32{});
                    Tensor sDKL2 = make_tensor(make_smem_ptr(smem->smem_mma_dk_lower_inter.data()),   SmemLayoutInputFP32{});
                    auto sDQ_sub   = local_tile(sDQ,   Shape<_16, Int<K_TILE>>{}, make_coord(warp_id_in_wg, _0{}));
                    auto sDQ2_sub  = local_tile(sDQ2,  Shape<_16, Int<K_TILE>>{}, make_coord(warp_id_in_wg, _0{}));
                    auto sDKL_sub  = local_tile(sDKL,  Shape<_16, Int<K_TILE>>{}, make_coord(warp_id_in_wg, _0{}));
                    auto sDKL2_sub = local_tile(sDKL2, Shape<_16, Int<K_TILE>>{}, make_coord(warp_id_in_wg, _0{}));
                    auto tCsDQ   = thr_mma.partition_C(sDQ_sub);
                    auto tCsDQ2  = thr_mma.partition_C(sDQ2_sub);
                    auto tCsDKL  = thr_mma.partition_C(sDKL_sub);
                    auto tCsDKL2 = thr_mma.partition_C(sDKL2_sub);
                    cute::copy(tDQ_acc,   tCsDQ);
                    cute::copy(tDQ2_acc,  tCsDQ2);
                    cute::copy(tDKL_acc,  tCsDKL);
                    cute::copy(tDKL2_acc, tCsDKL2);
                }

                fence_view_async_shared();
                cute::arrive_barrier(smem->bar_mma_dq_done);

                // === QKG PHASE: dKt (transposed A, into smem_mma_dq) ===
                cute::wait_barrier(smem->bar_qkg_all_ready, b_phase);
                // Wait for Prep to finish reading dQ/dQ2 before we overwrite smem_mma_dq with dKt.
                cute::wait_barrier(smem->bar_prep_dq_consumed, b_phase);

                FragC_t tDKT_intra_acc = partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{});
                FragC_t tDKT_inter_acc = partition_fragment_C(tiled_mma, Shape<_16, Int<K_TILE>>{});
                clear(tDKT_intra_acc);
                clear(tDKT_inter_acc);

                // A layout for dKt MMA: (M=16, K=32) where K=[dAqk^T 16 rows | dAkk^T 16 rows].
                // We build A by manual scalar SMEM read using identity-tensor coordinates,
                // because dAqk/dAkk must be transposed and stacked — no simple local_tile view.
                using AShape = Shape<_16, Int<2 * SUB_T_TILE>>;
                // Build a "dummy" smem-backed tensor solely to let partition_fragment_A
                // derive the per-thread register layout. We fill values manually below.
                auto sA_proto = local_tile(sDAqk, Shape<_16, Int<2 * SUB_T_TILE>>{}, make_coord(0, 0));

                // ---- QKG intra: 6 strict-off-diagonal (i, j) pairs (i > j) ----
                CUTE_UNROLL
                for (int s = 0; s < 6; ++s) {
                    if (warp_id_in_wg != qkg_intra_j[s]) continue;
                    int i_block = qkg_intra_i[s];
                    int j_block = qkg_intra_j[s];

                    auto cA   = make_identity_tensor(AShape{});
                    auto tAcA = thr_mma.partition_A(cA);
                    auto tArA = thr_mma.partition_fragment_A(sA_proto);
                    CUTE_UNROLL
                    for (int i = 0; i < size(tArA); ++i) {
                        int r = get<0>(tAcA(i));
                        int c = get<1>(tAcA(i));
                        int g_j = j_block * 16 + r;
                        int g_i;
                        float v;
                        if (c < 16) {
                            g_i = i_block * 16 + c;
                            v   = sDAqk(g_i, g_j);
                        } else {
                            g_i = i_block * 16 + (c - 16);
                            v   = sDAkk(g_i, g_j);
                        }
                        bool keep = (g_i < sub_seq_len) && (g_j < sub_seq_len);
                        tArA(i) = keep ? tfloat32_t(v) : tfloat32_t(0.0f);
                    }

                    Tensor sB = make_tensor(make_smem_ptr(smem->qkg_all.intra[s].data()),
                                            SmemLayoutMatBTF32Tranposed<2>{});
                    auto tBsB = thr_mma.partition_B(sB);
                    auto tBrB = thr_mma.partition_fragment_B(sB);
                    cute::copy(tBsB, tBrB);

                    cute::gemm(tiled_mma, tArA, tBrB, tDKT_intra_acc);
                }

                // ---- QKG inter: 4 diagonal blocks with causal mask ----
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s) {
                    if (warp_id_in_wg != s) continue;
                    int block = s;

                    auto cA   = make_identity_tensor(AShape{});
                    auto tAcA = thr_mma.partition_A(cA);
                    auto tArA = thr_mma.partition_fragment_A(sA_proto);
                    CUTE_UNROLL
                    for (int i = 0; i < size(tArA); ++i) {
                        int r = get<0>(tAcA(i));
                        int c = get<1>(tAcA(i));
                        int g_j = block * 16 + r;
                        int g_i;
                        float v;
                        if (c < 16) {
                            g_i = block * 16 + c;
                            v   = sDAqk(g_i, g_j);
                        } else {
                            g_i = block * 16 + (c - 16);
                            v   = sDAkk(g_i, g_j);
                        }
                        bool keep = (g_i >= g_j) && (g_i < sub_seq_len) && (g_j < sub_seq_len);
                        tArA(i) = keep ? tfloat32_t(v) : tfloat32_t(0.0f);
                    }

                    Tensor sB = make_tensor(make_smem_ptr(smem->qkg_all.inter[s].data()),
                                            SmemLayoutMatBTF32Tranposed<2>{});
                    auto tBsB = thr_mma.partition_B(sB);
                    auto tBrB = thr_mma.partition_fragment_B(sB);
                    cute::copy(tBsB, tBrB);

                    cute::gemm(tiled_mma, tArA, tBrB, tDKT_inter_acc);
                }

                // R2S: intra off-diag dkt → smem_mma_dq (lower half reads with next_top scale);
                //      inter diagonal dkt → smem_mma_dq2 (upper half reads with mid scale).
                {
                    Tensor sDKt_intra = make_tensor(make_smem_ptr(smem->smem_mma_dq.data()),  SmemLayoutInputFP32{});
                    Tensor sDKt_inter = make_tensor(make_smem_ptr(smem->smem_mma_dq2.data()), SmemLayoutInputFP32{});
                    auto sDKt_intra_sub = local_tile(sDKt_intra, Shape<_16, Int<K_TILE>>{}, make_coord(warp_id_in_wg, _0{}));
                    auto sDKt_inter_sub = local_tile(sDKt_inter, Shape<_16, Int<K_TILE>>{}, make_coord(warp_id_in_wg, _0{}));
                    auto tCsDKt_intra = thr_mma.partition_C(sDKt_intra_sub);
                    auto tCsDKt_inter = thr_mma.partition_C(sDKt_inter_sub);
                    cute::copy(tDKT_intra_acc, tCsDKt_intra);
                    cute::copy(tDKT_inter_acc, tCsDKt_inter);
                }

                fence_view_async_shared();
                cute::arrive_barrier(smem->bar_mma_dkt_done);

                b_phase ^= 1;
            }

            state_phase ^= 1 << (buf_idx_A + NUM_BUF);
            buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;

#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0 && thread_idx == 0) printf("[MMA] tile_idx=%d done\n", tile_idx);
#endif
        }

        // =================================================================
        // Prep warpgroups — scalar B-operand setup + epilogue
        // =================================================================
    } else {
        // cutlass::arch::warpgroup_reg_alloc<REG_PREP>();

        const int prep_thread = thread_idx - PREP_THREAD_OFFSET;  // 0..255
        const int prep_wg_idx = prep_thread / 128;                // 0 or 1
        const int idx_in_warpgroup = prep_thread % 128;
#if KDA_BWD_SM90_DEBUG_PRINT
        if (blockIdx.x == 0 && idx_in_warpgroup == 0) printf("[PREP WG%d] entered branch is_valid=%d\n", prep_wg_idx, (int)tile_scheduler.is_valid());
#endif
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int A_phase = (state_phase >> (buf_idx_A + NUM_BUF)) & 1;
#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0 && idx_in_warpgroup == 0 && prep_wg_idx == 0)
                printf("[PREP WG0] before wait bar_load_dA_ready buf=%d expect=%d\n", buf_idx_A, A_phase ^ 1);
#endif
            cute::wait_barrier(smem->bar_load_dA_ready[buf_idx_A], A_phase);
#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0 && idx_in_warpgroup == 0 && prep_wg_idx == 0)
                printf("[PREP WG0] after wait bar_load_dA_ready buf=%d\n", buf_idx_A);
#endif

            int tid = tile_scheduler.get_current_tile_id();
            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int start_offset = cu_len_ptr[batch_idx];
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

#if KDA_BWD_SM90_DEBUG_PRINT
            if (blockIdx.x == 0 && idx_in_warpgroup == 0 && prep_wg_idx == 0 && tile_idx == 0 &&
                batch_idx == 0 && head_idx == 0) {
                printf("[PREP] tile=%d batch=%d head=%d sub_seq_len=%d start_off=%d\n",
                       tile_idx, batch_idx, head_idx, sub_seq_len, start_offset);
            }
#endif

            if (prep_wg_idx == 0) {
                prep_compute_epilogue_body<0>(
                    smem, params, idx_in_warpgroup,
                    state_phase, buf_idx_A, buf_idx_value,
                    batch_idx, head_idx, tile_idx,
                    start_offset, sub_seq_len, A_phase);
            } else {
                prep_compute_epilogue_body<1>(
                    smem, params, idx_in_warpgroup,
                    state_phase, buf_idx_A, buf_idx_value,
                    batch_idx, head_idx, tile_idx,
                    start_offset, sub_seq_len, A_phase);
            }

            cute::arrive_barrier(smem->bar_dA_free[buf_idx_A]);
            state_phase ^= 1 << (buf_idx_A + NUM_BUF);
            buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
        }
    }
}

// =====================================================================
// Host Launch Function
// =====================================================================
void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream) {
    KDA_ASSERT(params.d % K_TILE == 0);
    int total_q_len = params.total_q_len;
    int H = params.h;
    int D = params.d;
    int BT = params.chunk_size;

    // ── Create TMA descriptors ──
    auto shape_QKG = make_shape(total_q_len, D, H);
    auto stride_QKG = make_stride(H * D, _1{}, D);

    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.q_ptr), make_layout(shape_QKG, stride_QKG)),
        SmemLayoutInputBF16{});

    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.k_ptr), make_layout(shape_QKG, stride_QKG)),
        SmemLayoutInputBF16{});

    auto tma_G = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.g_ptr), make_layout(shape_QKG, stride_QKG)),
        SmemLayoutInputFP32{});

    auto shape_DA = make_shape(total_q_len, BT, H);
    auto stride_DA = make_stride(H * BT, _1{}, BT);

    auto tma_DAqk = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.dAqk_ptr), make_layout(shape_DA, stride_DA)),
        SmemLayoutDA{});

    auto tma_DAkk = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.dAkk_ptr), make_layout(shape_DA, stride_DA)),
        SmemLayoutDA{});

    auto tma_DQ = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.dq_ptr), make_layout(shape_QKG, stride_QKG)),
        SmemLayoutInputFP32{});

    auto tma_DK = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.dk_ptr), make_layout(shape_QKG, stride_QKG)),
        SmemLayoutInputFP32{});

    auto tma_DG = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.dg_ptr), make_layout(shape_QKG, stride_QKG)),
        SmemLayoutInputFP32{});

    auto tma_params_val = TmaParams<
        decltype(shape_QKG),
        decltype(shape_DA),
        decltype(tma_Q),
        decltype(tma_K),
        decltype(tma_G),
        decltype(tma_DAqk),
        decltype(tma_DAkk),
        decltype(tma_DQ),
        decltype(tma_DK),
        decltype(tma_DG)>{
        shape_QKG,
        shape_DA,
        tma_Q,
        tma_K,
        tma_G,
        tma_DAqk,
        tma_DAkk,
        tma_DQ,
        tma_DK,
        tma_DG,
    };

    auto kda_kernel = &kda_bwd_intra_sm90_kernel<decltype(tma_params_val)>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    static bool printed = false;
    if (!printed) {
        printf("[HOST] SharedStorage size = %zu bytes (%.1f KB)\n", smem_size, smem_size/1024.0);
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64ull * 1024 * 1024);
        printed = true;
    }
    CHECK_CUDA(cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(NUM_THREADS, 1, 1);
    kda_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params_val);
}

}  // namespace sm90_bwd

// =====================================================================
// C API — exposed for standalone compilation / FFI
// =====================================================================
extern "C" void
launch_c_kda_bwd_intra_sm90(void* params, cudaStream_t stream) {
    sm90_bwd::run_kda_bwd_intra_sm90(*static_cast<KDA_bwd_intra_params*>(params), stream);
}
