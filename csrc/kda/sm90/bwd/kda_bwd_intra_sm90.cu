#include <math.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

using namespace cute;

#include "kda/sm90/bwd/kda_config.h"

// ============================================================
// Constants
// ============================================================
static constexpr int BC = 16;
static constexpr int BK = 32;
static constexpr int BT = 64;
static constexpr int NC = 4;  // BT / BC
static constexpr int NK = 4;  // K / BK (K=128 assumed)
static constexpr int NUM_THREADS = 128;

// ============================================================
// CuTe type aliases
// ============================================================

// SM80 TF32 MMA: 16x8x8, TN layout. Tiled across 4 warps in N.
using MMA_Atom_TF32 = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
using TiledMMA_t = TiledMMA<MMA_Atom_TF32, Layout<Shape<_1, _4, _1>>>;
// Tiled shape: M=16, N=32, K=8. For K=16 we iterate k=0,1.

// SMEM layouts (row-major, col stride-1)
// [16,32] bf16, row_bytes=64B → Swizzle<2,3,3>, loaded via cp.async
using SmemLayoutQK = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));
// [16,32] fp32, row_bytes=128B → Swizzle<3,2,3> (128B mode), loaded via cooperative copy
using SmemLayoutG = decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));
// [16,16] fp32, row_bytes=64B → Swizzle<2,2,3>, loaded via cp.async (Phase 1) / cooperative (Phase 2)
using SmemLayoutDA = decltype(composition(Swizzle<2, 2, 3>{}, Layout<Shape<Int<BC>, Int<BC>>, Stride<Int<BC>, _1>>{}));
// [32,16] fp32, padded stride 17 → gcd(17,32)=1 → zero bank conflicts for both row/col access
using SmemLayoutB_op = Layout<Shape<Int<BK>, Int<BC>>, Stride<Int<BC + 1>, _1>>;
// [16,32] fp32, row_bytes=128B → Swizzle<3,2,3> (128B mode)
using SmemLayoutAcc = decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));

// S2R atoms for MMA operand loads
// A (dA): ldmatrix.x4 — 1 warp instruction loads [16,8] floats (4 regs/thread)
using S2RAtomA = Copy_Atom<SM75_U32x4_LDSM_N, float>;
// B (KG): scalar shared load — stride-17 padding gives 0 bank conflicts
using S2RAtomB = Copy_Atom<UniversalCopy<float>, float>;

// ============================================================
// Shared memory
// ============================================================
struct SmemStorage {
    array_aligned<__nv_bfloat16, cosize_v<SmemLayoutQK>, 128> s_q;  // [16,32] persistent bf16
    array_aligned<__nv_bfloat16, cosize_v<SmemLayoutQK>, 128> s_k;  // [16,32] persistent bf16
    array_aligned<float, cosize_v<SmemLayoutG>, 128> s_g;           // [16,32] persistent fp32
    array_aligned<float, BC> s_beta;                           // [16]    persistent
    array_aligned<float, BK> s_gn;                             // [32]    gate anchor
    array_aligned<float, cosize_v<SmemLayoutDA>, 128> s_dA_qk;  // [16,16] per-iter
    array_aligned<float, cosize_v<SmemLayoutDA>, 128> s_dA_kk;  // [16,16] per-iter
    array_aligned<float, cosize_v<SmemLayoutB_op>> s_KG;       // [32,16] Phase1: KG, Phase2: QG
    union {
        array_aligned<float, cosize_v<SmemLayoutB_op>> s_KBG;  // Phase 2: KBG operand
        array_aligned<float, cosize_v<SmemLayoutAcc>> s_acc;  // Phase 1: db reduction scratch
    };
    array_aligned<float, BC> s_db;  // [16] db output
};

// ============================================================
// MMA helper: acc += A[16,16] @ B[32,16]^T → C[16,32]
// Follows sgemm_sm80 pattern: s2r copy with retile_D
// ============================================================
__device__ __forceinline__ void
gemm_m16n32k16(const float* s_A, const float* s_Bop, float acc[4], int tid) {
    TiledMMA_t tiled_mma;
    auto sA = make_tensor(make_smem_ptr(s_A), SmemLayoutDA{});      // (16,16)
    auto sB = make_tensor(make_smem_ptr(s_Bop), SmemLayoutB_op{});  // (32,16)

    auto thr_mma = tiled_mma.get_slice(tid);
    auto tCrA = thr_mma.partition_fragment_A(sA);  // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(sB);  // (MMA, MMA_N, MMA_K)

    // Accumulator: create using a [16,32] shape for C
    auto sC_dummy = make_tensor(make_smem_ptr(s_A), SmemLayoutAcc{});
    auto tCrC = thr_mma.partition_fragment_C(sC_dummy);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = acc[i];
    }

    auto s2r_copy_a = make_tiled_copy_A(S2RAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RAtomB{}, tiled_mma);
    auto thr_s2r_a = s2r_copy_a.get_slice(tid);
    auto thr_s2r_b = s2r_copy_b.get_slice(tid);

    auto tXsA = thr_s2r_a.partition_S(sA);
    auto tXrA = thr_s2r_a.retile_D(tCrA);
    auto tXsB = thr_s2r_b.partition_S(sB);
    auto tXrB = thr_s2r_b.retile_D(tCrB);

    // K-loop: K=16 / atom_K=8 = 2 iterations
    auto K_BLOCK_MAX = size<2>(tCrA);
    CUTE_UNROLL
    for (int k = 0; k < K_BLOCK_MAX; ++k) {
        copy(s2r_copy_a, tXsA(_, _, k), tXrA(_, _, k));
        copy(s2r_copy_b, tXsB(_, _, k), tXrB(_, _, k));
        gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), tCrC);
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        acc[i] = tCrC(i);
    }
}

// ============================================================
// Fused dual-GEMM with shared B:
//   acc1 += A1 @ B^T,  acc2 += A2 @ B^T
// B operand S2R load done once per k-step, saving half B bandwidth.
// ============================================================
__device__ __forceinline__ void
gemm_m16n32k16_shared_b(
    const float* s_A1, const float* s_A2,
    const float* s_Bop,
    float acc1[4], float acc2[4], int tid)
{
    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto sB = make_tensor(make_smem_ptr(s_Bop), SmemLayoutB_op{});
    auto sA1 = make_tensor(make_smem_ptr(s_A1), SmemLayoutDA{});
    auto sA2 = make_tensor(make_smem_ptr(s_A2), SmemLayoutDA{});

    auto tCrB  = thr_mma.partition_fragment_B(sB);
    auto tCrA1 = thr_mma.partition_fragment_A(sA1);
    auto tCrA2 = thr_mma.partition_fragment_A(sA2);

    auto sC_dummy = make_tensor(make_smem_ptr(s_A1), SmemLayoutAcc{});
    auto tCrC1 = thr_mma.partition_fragment_C(sC_dummy);
    auto tCrC2 = thr_mma.partition_fragment_C(sC_dummy);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC1); ++i) { tCrC1(i) = acc1[i]; tCrC2(i) = acc2[i]; }

    auto s2r_copy_a = make_tiled_copy_A(S2RAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RAtomB{}, tiled_mma);
    auto thr_s2r_a = s2r_copy_a.get_slice(tid);
    auto thr_s2r_b = s2r_copy_b.get_slice(tid);

    auto tXsB  = thr_s2r_b.partition_S(sB);
    auto tXrB  = thr_s2r_b.retile_D(tCrB);
    auto tXsA1 = thr_s2r_a.partition_S(sA1);
    auto tXrA1 = thr_s2r_a.retile_D(tCrA1);
    auto tXsA2 = thr_s2r_a.partition_S(sA2);
    auto tXrA2 = thr_s2r_a.retile_D(tCrA2);

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA1); ++k) {
        copy(s2r_copy_b, tXsB(_, _, k), tXrB(_, _, k));   // B loaded once
        copy(s2r_copy_a, tXsA1(_, _, k), tXrA1(_, _, k));
        gemm(tiled_mma, tCrA1(_, _, k), tCrB(_, _, k), tCrC1);
        copy(s2r_copy_a, tXsA2(_, _, k), tXrA2(_, _, k));
        gemm(tiled_mma, tCrA2(_, _, k), tCrB(_, _, k), tCrC2);
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC1); ++i) { acc1[i] = tCrC1(i); acc2[i] = tCrC2(i); }
}

__device__ __forceinline__ __nv_bfloat162 bitcast_bf162(uint32_t x) {
    return reinterpret_cast<__nv_bfloat162&>(x);
}

// ============================================================
// Per-thread accumulator → (row, col) mapping
// SM80_16x8x8 CLayout = SM80_16x8_Row (column-major in MxN tile):
//   pos = m + n * 16 (NOT row-major m * 8 + n)
//   PTX m16n8k8 f32 mapping:
//     GroupID = lane / 4 → base row
//     ThreadInGroup = lane % 4 → base col pair
//     reg[v]: row = GroupID + (v/2)*8, col = ThreadInGroup*2 + (v%2)
// ============================================================
__device__ __forceinline__ void
get_acc_row_col(int tid, int v, int& row, int& col) {
    int lane = tid % 32;
    int warp_id = tid / 32;
    row = (lane / 4) + (v / 2) * 8;
    col = (lane % 4) * 2 + (v % 2) + warp_id * 8;
}

// Per-thread "pair" base position (rh in {0,1}): two adjacent fp32 acc values
// (v=2*rh, v=2*rh+1) live at (row, col_base..col_base+1). Use this to issue
// 8B-aligned float2 LDS/STS in epilogues — replaces 4 scalar scatter writes.
__device__ __forceinline__ void
get_acc_pair_row_col(int tid, int rh, int& row, int& col_base) {
    int lane = tid % 32;
    int warp_id = tid / 32;
    row = (lane / 4) + rh * 8;
    col_base = (lane % 4) * 2 + warp_id * 8;
}

// ============================================================
// Main kernel
// ============================================================
__global__ void
__launch_bounds__(NUM_THREADS) kda_bwd_intra_kernel_sm90(__grid_constant__ const KDA_bwd_intra_params params) {
    extern __shared__ char smem_buf[];
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(smem_buf);

    // ── Extract typed pointers (active code path only) ──
    const auto* beta_ptr = reinterpret_cast<const __nv_bfloat16*>(params.beta_ptr);
    auto* dq_out_ptr = reinterpret_cast<__nv_bfloat16*>(params.dq_out_ptr);
    auto* dk_out_ptr = reinterpret_cast<__nv_bfloat16*>(params.dk_out_ptr);
    auto* db_out_ptr = reinterpret_cast<float*>(params.db_out_ptr);
    auto* dg_out_ptr = reinterpret_cast<float*>(params.dg_out_ptr);
    const auto* k_ptr = reinterpret_cast<const __nv_bfloat16*>(params.k_ptr);
    const auto* g_ptr = reinterpret_cast<const float*>(params.g_ptr);
    const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q_ptr);
    const auto* dq_ptr = reinterpret_cast<const float*>(params.dq_ptr);
    const auto* dk_ptr = reinterpret_cast<const float*>(params.dk_ptr);
    const auto* dg_ptr = reinterpret_cast<const float*>(params.dg_ptr);
    const auto* dAqk_ptr = reinterpret_cast<const float*>(params.dAqk_ptr);
    const auto* dAkk_ptr = reinterpret_cast<const float*>(params.dAkk_ptr);
    const auto* cu_seqlens = reinterpret_cast<const int*>(params.cu_seqlens_ptr);
    const auto* chunk_idx = reinterpret_cast<const int*>(params.chunk_indices_ptr);

    const int H = params.h;
    const int K = params.d;
    const int total_q_len = params.total_q_len;

    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;

    // Decode block coordinates via tile scheduler
    NaiveTileScheduler scheduler(params.tile_scheduler_params);
    auto coord = scheduler.get_block_coord(chunk_idx, cu_seqlens);
    const int i_h = get<1>(coord);
    const int i_t = get<2>(coord);
    const int i_k = get<3>(coord);
    const int i_i = get<4>(coord);
    const int bos = get<5>(coord);
    const int T_seq = get<6>(coord);

    const int i_ti = i_t * BT + i_i * BC;
    if (i_ti >= T_seq)
        return;

    const int tile_row = i_t * NC + i_i;

    // ── CuTe gmem tensors ──
    auto make_seq_hd = [&](auto ptr, int D) {
        auto g_full = make_tensor(make_gmem_ptr(ptr), make_shape(total_q_len, H, D), make_stride(H * D, D, _1{}));
        auto g_head = g_full(_, i_h, _);
        return make_tensor(g_head.data() + g_head.layout()(bos, 0), make_shape(T_seq, D), stride(g_head));
    };

    auto mDqOut = make_seq_hd(dq_out_ptr, K);
    auto mDkOut = make_seq_hd(dk_out_ptr, K);
    auto mDgOut = make_seq_hd(dg_out_ptr, K);
    auto mBeta = make_tensor(make_gmem_ptr(beta_ptr + bos * H + i_h), make_shape(T_seq), make_stride(H));
    auto mDBout = make_tensor(
        make_gmem_ptr(db_out_ptr + i_k * total_q_len * H + bos * H + i_h), make_shape(T_seq), make_stride(H));

    auto mQ = make_seq_hd(q_ptr, K);
    auto mK = make_seq_hd(k_ptr, K);
    auto mG = make_seq_hd(g_ptr, K);
    auto mDq = make_seq_hd(dq_ptr, K);
    auto mDk = make_seq_hd(dk_ptr, K);
    auto mDg = make_seq_hd(dg_ptr, K);
    auto mDAqk_g = make_seq_hd(dAqk_ptr, BT);  // [T_seq, BT] for cp.async Phase 1
    auto mDAkk_g = make_seq_hd(dAkk_ptr, BT);

    auto tile_hk = make_shape(Int<BC>{}, Int<BK>{});
    auto tile_da = make_shape(Int<BC>{}, Int<BC>{});  // [16,16] for dA tiles

    // ── SMEM tensor views ──
    auto sQ = make_tensor(make_smem_ptr(smem.s_q.data()), SmemLayoutQK{});
    auto sK = make_tensor(make_smem_ptr(smem.s_k.data()), SmemLayoutQK{});
    auto sG = make_tensor(make_smem_ptr(smem.s_g.data()), SmemLayoutG{});
    auto sKG = make_tensor(make_smem_ptr(smem.s_KG.data()), SmemLayoutB_op{});  // Phase1: KG, Phase2: QG
    auto sQG = sKG;  // alias — same physical smem, different semantic name
    auto sKBG = make_tensor(make_smem_ptr(smem.s_KBG.data()), SmemLayoutB_op{});
    auto sDAqk = make_tensor(make_smem_ptr(smem.s_dA_qk.data()), SmemLayoutDA{});
    auto sDAkk = make_tensor(make_smem_ptr(smem.s_dA_kk.data()), SmemLayoutDA{});

    // Whether this tile touches the sequence boundary (last few rows may be OOB)
    const bool is_boundary = (i_ti + BC) > T_seq;

    // ── Load persistent tiles via cp.async: Q, K, G + cooperative beta ──
    {
        auto gQ_tile = local_tile(mQ, tile_hk, make_coord(tile_row, i_k));
        auto gK_tile = local_tile(mK, tile_hk, make_coord(tile_row, i_k));
        auto gG_tile = local_tile(mG, tile_hk, make_coord(tile_row, i_k));
        auto gBeta_tile = local_tile(mBeta, Int<BC>{}, tile_row);

        if (is_boundary) {
            // Slow path: boundary tile — use src_size=0 to zero-fill OOB rows
            constexpr int BF16_CHUNKS = (BC * BK) / 8;
            for (int ci = tid; ci < BF16_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 8;
                int r = elem / BK, c = elem % BK;
                int src_size = (i_ti + r >= T_seq) ? 0 : 16;
                uint32_t dstQ = cute::cast_smem_ptr_to_uint(&sQ(r, c));
                uint32_t dstK = cute::cast_smem_ptr_to_uint(&sK(r, c));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dstQ), "l"(&gQ_tile(r, c)), "r"(src_size));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dstK), "l"(&gK_tile(r, c)), "r"(src_size));
            }
            constexpr int FP32_CHUNKS = (BC * BK) / 4;
            for (int ci = tid; ci < FP32_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 4;
                int r = elem / BK, c = elem % BK;
                int src_size = (i_ti + r >= T_seq) ? 0 : 16;
                uint32_t dstG = cute::cast_smem_ptr_to_uint(&sG(r, c));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dstG), "l"(&gG_tile(r, c)), "r"(src_size));
            }
            if (tid < BC) {
                smem.s_beta[tid] = (i_ti + tid < T_seq) ? __bfloat162float(gBeta_tile(tid)) : 0.f;
            }
        } else {
            // Fast path: no boundary — unconditional 3-operand cp.async
            constexpr int BF16_CHUNKS = (BC * BK) / 8;
            for (int ci = tid; ci < BF16_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 8;
                int r = elem / BK, c = elem % BK;
                uint32_t dstQ = cute::cast_smem_ptr_to_uint(&sQ(r, c));
                uint32_t dstK = cute::cast_smem_ptr_to_uint(&sK(r, c));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dstQ), "l"(&gQ_tile(r, c)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dstK), "l"(&gK_tile(r, c)));
            }
            constexpr int FP32_CHUNKS = (BC * BK) / 4;
            for (int ci = tid; ci < FP32_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 4;
                int r = elem / BK, c = elem % BK;
                uint32_t dstG = cute::cast_smem_ptr_to_uint(&sG(r, c));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dstG), "l"(&gG_tile(r, c)));
            }
            if (tid < BC) {
                smem.s_beta[tid] = __bfloat162float(gBeta_tile(tid));
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    // Per-thread accumulators (4 values per thread, covering 16×32 output tile)
    float dq2_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float dk2_acc[4] = {0.f, 0.f, 0.f, 0.f};

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1 Off-diagonal (j < i_i): dQ += dAqk × KG, dK += dAkk × KG
    // ════════════════════════════════════════════════════════════════════════
    if (i_i > 0) {
        if (tid < BK) {
            smem.s_gn[tid] = sG(0, tid);
        }
        __syncthreads();

        #pragma unroll 1
        for (int i_j = 0; i_j < i_i; ++i_j) {
            int j_tile = i_t * NC + i_j;

            // cp.async load dAqk + dAkk: fp32 [16,16] = 256 elems, 4 per chunk → 64 chunks
            {
                auto gDAqk_j = local_tile(mDAqk_g, tile_da, make_coord(tile_row, i_j));
                auto gDAkk_j = local_tile(mDAkk_g, tile_da, make_coord(tile_row, i_j));
                constexpr int DA_CHUNKS = (BC * BC) / 4;  // 64
                if (is_boundary) {
                    for (int ci = tid; ci < DA_CHUNKS; ci += NUM_THREADS) {
                        int elem = ci * 4;
                        int r = elem / BC, c = elem % BC;
                        uint32_t dst_qk = cute::cast_smem_ptr_to_uint(&sDAqk(r, c));
                        uint32_t dst_kk = cute::cast_smem_ptr_to_uint(&sDAkk(r, c));
                        int src_size = (i_ti + r >= T_seq) ? 0 : 16;
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_qk), "l"(&gDAqk_j(r, c)), "r"(src_size));
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_kk), "l"(&gDAkk_j(r, c)), "r"(src_size));
                    }
                } else {
                    for (int ci = tid; ci < DA_CHUNKS; ci += NUM_THREADS) {
                        int elem = ci * 4;
                        int r = elem / BC, c = elem % BC;
                        uint32_t dst_qk = cute::cast_smem_ptr_to_uint(&sDAqk(r, c));
                        uint32_t dst_kk = cute::cast_smem_ptr_to_uint(&sDAkk(r, c));
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_qk), "l"(&gDAqk_j(r, c)));
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_kk), "l"(&gDAkk_j(r, c)));
                    }
                }
            }

            // Build KG operand from gmem using 4-wide vector loads along contiguous N
            {
                auto gK_j = local_tile(mK, tile_hk, make_coord(j_tile, i_k));
                auto gG_j = local_tile(mG, tile_hk, make_coord(j_tile, i_k));
                constexpr int VEC_ELEMS = BC * BK / 4;  // 128
                for (int vi = tid; vi < VEC_ELEMS; vi += NUM_THREADS) {
                    int k_dim = (vi * 4) / BK;
                    int n = (vi * 4) % BK;

                    uint2 k_pack = *reinterpret_cast<const uint2*>(&gK_j(k_dim, n));
                    __nv_bfloat162 k01 = bitcast_bf162(k_pack.x);
                    __nv_bfloat162 k23 = bitcast_bf162(k_pack.y);
                    float2 kf01 = __bfloat1622float2(k01);
                    float2 kf23 = __bfloat1622float2(k23);
                    float4 gv = *reinterpret_cast<const float4*>(&gG_j(k_dim, n));

                    sKG(n + 0, k_dim) = kf01.x * exp2f(smem.s_gn[n + 0] - gv.x);
                    sKG(n + 1, k_dim) = kf01.y * exp2f(smem.s_gn[n + 1] - gv.y);
                    sKG(n + 2, k_dim) = kf23.x * exp2f(smem.s_gn[n + 2] - gv.z);
                    sKG(n + 3, k_dim) = kf23.y * exp2f(smem.s_gn[n + 3] - gv.w);
                }
            }

            // Wait for cp.async dA + fence KG writes
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();

            gemm_m16n32k16_shared_b(smem.s_dA_qk.data(), smem.s_dA_kk.data(), smem.s_KG.data(), dq2_acc, dk2_acc, tid);
            __syncthreads();
        }

        // Post-multiply: *= exp2(g_i - gn)
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            float scale = exp2f(sG(row, col) - smem.s_gn[col]);
            dq2_acc[v] *= scale;
            dk2_acc[v] *= scale;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1 Diagonal (j == i_i, lower-triangular): dQ, dK
    // ════════════════════════════════════════════════════════════════════════
    __syncthreads();
    {
        int mid = min(BC / 2, T_seq - i_ti - 1);
        if (tid < BK) {
            smem.s_gn[tid] = sG(mid, tid);
        }
        __syncthreads();

        // cp.async load dAqk + dAkk diagonal tile
        {
            auto gDAqk_diag = local_tile(mDAqk_g, tile_da, make_coord(tile_row, i_i));
            auto gDAkk_diag = local_tile(mDAkk_g, tile_da, make_coord(tile_row, i_i));
            constexpr int DA_CHUNKS = (BC * BC) / 4;  // 64
            if (is_boundary) {
                for (int ci = tid; ci < DA_CHUNKS; ci += NUM_THREADS) {
                    int elem = ci * 4;
                    int r = elem / BC, c = elem % BC;
                    uint32_t dst_qk = cute::cast_smem_ptr_to_uint(&sDAqk(r, c));
                    uint32_t dst_kk = cute::cast_smem_ptr_to_uint(&sDAkk(r, c));
                    int src_size = (i_ti + r >= T_seq) ? 0 : 16;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_qk), "l"(&gDAqk_diag(r, c)), "r"(src_size));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_kk), "l"(&gDAkk_diag(r, c)), "r"(src_size));
                }
            } else {
                for (int ci = tid; ci < DA_CHUNKS; ci += NUM_THREADS) {
                    int elem = ci * 4;
                    int r = elem / BC, c = elem % BC;
                    uint32_t dst_qk = cute::cast_smem_ptr_to_uint(&sDAqk(r, c));
                    uint32_t dst_kk = cute::cast_smem_ptr_to_uint(&sDAkk(r, c));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_qk), "l"(&gDAqk_diag(r, c)));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_kk), "l"(&gDAkk_diag(r, c)));
                }
            }
        }

        // Build KG from persistent smem K, G (overlaps TMA)
        for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
            int n = idx % BK;
            int k_dim = idx / BK;
            bool valid = (i_ti + k_dim) < T_seq;
            float g_diff = valid ? (sG(k_dim, n) - smem.s_gn[n]) : 0.f;
            sKG(n, k_dim) = valid ? (__bfloat162float(sK(k_dim, n)) * exp2f(-g_diff)) : 0.f;
        }

        // Wait for cp.async dA + fence KG writes
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Apply lower-triangular mask to both dAqk and dAkk
        for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
            int r = idx / BC, c = idx % BC;
            bool valid = (r >= c) && (i_ti + r < T_seq) && (i_ti + c < T_seq);
            if (!valid) {
                sDAqk(r, c) = 0.f;
                sDAkk(r, c) = 0.f;
            }
        }
        __syncthreads();

        float tmp_dq[4] = {0.f, 0.f, 0.f, 0.f};
        float tmp_dk[4] = {0.f, 0.f, 0.f, 0.f};
        gemm_m16n32k16_shared_b(smem.s_dA_qk.data(), smem.s_dA_kk.data(), smem.s_KG.data(), tmp_dq, tmp_dk, tid);

        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            bool valid = (i_ti + row) < T_seq;
            float g_diff = valid ? (sG(row, col) - smem.s_gn[col]) : 0.f;
            float scale = valid ? exp2f(g_diff) : 0.f;
            dq2_acc[v] += tmp_dq[v] * scale;
            dk2_acc[v] += tmp_dk[v] * scale;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // INTERMEDIATE: db = reduce(dk2 * k), dk2 *= beta
    // ════════════════════════════════════════════════════════════════════════
    __syncthreads();
    {
        auto sAcc = make_tensor(make_smem_ptr(smem.s_acc.data()), SmemLayoutAcc{});
        // Write dk2 * k into smem for row-reduction
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            float kv = __bfloat162float(sK(row, col));
            sAcc(row, col) = dk2_acc[v] * kv;
        }
        __syncthreads();

        // Row reduction: 128 threads → 16 rows, 8 threads/row, 4 cols each
        {
            int red_row = tid / 8;
            int red_lane = tid % 8;
            float sum = 0.f;
#pragma unroll
            for (int c = red_lane * 4; c < red_lane * 4 + 4; ++c) {
                sum += sAcc(red_row, c);
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            if (red_lane == 0) {
                smem.s_db[red_row] = sum;
            }
        }

        // dk2 *= beta
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            dk2_acc[v] *= smem.s_beta[row];
        }
        __syncthreads();
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1 EPILOGUE: store dQ output, db output (coalesced via smem staging)
    // ════════════════════════════════════════════════════════════════════════
    {
        auto gDq_tile = local_tile(mDq, tile_hk, make_coord(tile_row, i_k));
        auto gDqOut_tile = local_tile(mDqOut, tile_hk, make_coord(tile_row, i_k));
        auto sStage = make_tensor(make_smem_ptr(smem.s_acc.data()), SmemLayoutAcc{});

        // Preload old dq row-major so the hot dq_prev reads avoid accumulator-mapped gmem gathers.
        {
            constexpr int VEC_ELEMS = BC * BK / 4;  // 128
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
        }
        __syncthreads();

        // Scatter: each thread computes final dq value and writes to smem (vec float2)
        for (int rh = 0; rh < 2; ++rh) {
            int row, col_base;
            get_acc_pair_row_col(tid, rh, row, col_base);
            float2* p = reinterpret_cast<float2*>(&sStage(row, col_base));
            float2 cur = *p;
            cur.x += dq2_acc[rh * 2 + 0];
            cur.y += dq2_acc[rh * 2 + 1];
            *p = cur;
        }
        __syncthreads();

        // Vectorized store to gmem as bf16x4 (64-bit)
        {
            int vi = tid;
            int r = (vi * 4) / BK;
            int c = (vi * 4) % BK;
            if ((i_ti + r) < T_seq) {
                __nv_bfloat162 lo = __floats2bfloat162_rn(sStage(r, c + 0), sStage(r, c + 1));
                __nv_bfloat162 hi = __floats2bfloat162_rn(sStage(r, c + 2), sStage(r, c + 3));
                uint2 packed;
                packed.x = reinterpret_cast<uint32_t&>(lo);
                packed.y = reinterpret_cast<uint32_t&>(hi);
                *reinterpret_cast<uint2*>(&gDqOut_tile(r, c)) = packed;
            }
        }
        if (tid < BC && (i_ti + tid) < T_seq) {
            mDBout(i_ti + tid) = smem.s_db[tid];
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2: dKT computation (transposed dA contributions)
    // ════════════════════════════════════════════════════════════════════════
    float dkt_acc[4] = {0.f, 0.f, 0.f, 0.f};
    __syncthreads();

    int NC_eff = min(NC, (T_seq - i_t * BT + BC - 1) / BC);

    // ── Phase 2 off-diagonal (j > i_i) ──
    if (i_i < NC_eff - 1) {
        int last_local = min(BC, T_seq - i_ti) - 1;
        if (tid < BK) {
            smem.s_gn[tid] = sG(last_local, tid);
        }
        __syncthreads();

        #pragma unroll 1
        for (int i_j = i_i + 1; i_j < NC_eff; ++i_j) {
            int j_tile = i_t * NC + i_j;
            int j_ti = i_t * BT + i_j * BC;

            // Coalesced transposed loads: read dA row-major as float4, transpose on smem write
            constexpr int DA_VEC_ELEMS = BC * BC / 4;  // 64
            for (int vi = tid; vi < DA_VEC_ELEMS; vi += NUM_THREADS) {
                int r = vi / (BC / 4);
                int c = (vi % (BC / 4)) * 4;
                bool valid = (j_ti + r) < T_seq;
                int gmem_addr = (bos + j_ti + r) * (H * BT) + i_h * BT + i_i * BC + c;
                if (valid) {
                    float4 qv = *reinterpret_cast<const float4*>(dAqk_ptr + gmem_addr);
                    float4 kv = *reinterpret_cast<const float4*>(dAkk_ptr + gmem_addr);
                    sDAqk(c + 0, r) = qv.x;
                    sDAqk(c + 1, r) = qv.y;
                    sDAqk(c + 2, r) = qv.z;
                    sDAqk(c + 3, r) = qv.w;
                    sDAkk(c + 0, r) = kv.x;
                    sDAkk(c + 1, r) = kv.y;
                    sDAkk(c + 2, r) = kv.z;
                    sDAkk(c + 3, r) = kv.w;
                } else {
                    sDAqk(c + 0, r) = 0.f;
                    sDAqk(c + 1, r) = 0.f;
                    sDAqk(c + 2, r) = 0.f;
                    sDAqk(c + 3, r) = 0.f;
                    sDAkk(c + 0, r) = 0.f;
                    sDAkk(c + 1, r) = 0.f;
                    sDAkk(c + 2, r) = 0.f;
                    sDAkk(c + 3, r) = 0.f;
                }
            }

            // Build QG = q_j * exp2(g_j - gn) and KBG = k_j * beta_j * exp2(g_j - gn)
            // using 4-wide vector loads along contiguous N
            {
                auto gQ_j = local_tile(mQ, tile_hk, make_coord(j_tile, i_k));
                auto gK_j = local_tile(mK, tile_hk, make_coord(j_tile, i_k));
                auto gG_j = local_tile(mG, tile_hk, make_coord(j_tile, i_k));
                auto gBeta_j = local_tile(mBeta, Int<BC>{}, j_tile);
                constexpr int VEC_ELEMS = BC * BK / 4;  // 128
                for (int vi = tid; vi < VEC_ELEMS; vi += NUM_THREADS) {
                    int k_dim = (vi * 4) / BK;
                    int n = (vi * 4) % BK;
                    bool valid = (j_ti + k_dim) < T_seq;
                    float bv = valid ? __bfloat162float(gBeta_j(k_dim)) : 0.f;

                    if (valid) {
                        uint2 q_pack = *reinterpret_cast<const uint2*>(&gQ_j(k_dim, n));
                        uint2 k_pack = *reinterpret_cast<const uint2*>(&gK_j(k_dim, n));
                        __nv_bfloat162 q01 = bitcast_bf162(q_pack.x);
                        __nv_bfloat162 q23 = bitcast_bf162(q_pack.y);
                        __nv_bfloat162 k01 = bitcast_bf162(k_pack.x);
                        __nv_bfloat162 k23 = bitcast_bf162(k_pack.y);
                        float2 qf01 = __bfloat1622float2(q01);
                        float2 qf23 = __bfloat1622float2(q23);
                        float2 kf01 = __bfloat1622float2(k01);
                        float2 kf23 = __bfloat1622float2(k23);
                        float4 gv = *reinterpret_cast<const float4*>(&gG_j(k_dim, n));

                        float gate0 = exp2f(gv.x - smem.s_gn[n + 0]);
                        float gate1 = exp2f(gv.y - smem.s_gn[n + 1]);
                        float gate2 = exp2f(gv.z - smem.s_gn[n + 2]);
                        float gate3 = exp2f(gv.w - smem.s_gn[n + 3]);

                        sKG(n + 0, k_dim) = qf01.x * gate0;
                        sKG(n + 1, k_dim) = qf01.y * gate1;
                        sKG(n + 2, k_dim) = qf23.x * gate2;
                        sKG(n + 3, k_dim) = qf23.y * gate3;

                        sKBG(n + 0, k_dim) = kf01.x * bv * gate0;
                        sKBG(n + 1, k_dim) = kf01.y * bv * gate1;
                        sKBG(n + 2, k_dim) = kf23.x * bv * gate2;
                        sKBG(n + 3, k_dim) = kf23.y * bv * gate3;
                    } else {
                        sKG(n + 0, k_dim) = 0.f;
                        sKG(n + 1, k_dim) = 0.f;
                        sKG(n + 2, k_dim) = 0.f;
                        sKG(n + 3, k_dim) = 0.f;
                        sKBG(n + 0, k_dim) = 0.f;
                        sKBG(n + 1, k_dim) = 0.f;
                        sKBG(n + 2, k_dim) = 0.f;
                        sKBG(n + 3, k_dim) = 0.f;
                    }
                }
            }
            __syncthreads();

            gemm_m16n32k16(smem.s_dA_qk.data(), smem.s_KG.data(), dkt_acc, tid);
            gemm_m16n32k16(smem.s_dA_kk.data(), smem.s_KBG.data(), dkt_acc, tid);
            __syncthreads();
        }

        // Post-multiply: dkt *= exp2(gn - g_i)
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            float scale = (!is_boundary || (i_ti + row) < T_seq) ? exp2f(smem.s_gn[col] - sG(row, col)) : 0.f;
            dkt_acc[v] *= scale;
        }
    }

    // ── Phase 2 diagonal (j == i_i, upper-triangular) ──
    __syncthreads();
    {
        int mid = min(BC / 2, T_seq - i_ti - 1);
        if (tid < BK) {
            smem.s_gn[tid] = sG(mid, tid);
        }
        __syncthreads();

        // Coalesced transposed loads with upper-tri mask.
        // Only vectorize the non-boundary case; boundary needs per-element guards.
        if (!is_boundary) {
            constexpr int DA_VEC_ELEMS = BC * BC / 4;  // 64
            for (int vi = tid; vi < DA_VEC_ELEMS; vi += NUM_THREADS) {
                int r = vi / (BC / 4);
                int c = (vi % (BC / 4)) * 4;
                int gmem_addr = (bos + i_ti + r) * (H * BT) + i_h * BT + i_i * BC + c;
                float4 qv = *reinterpret_cast<const float4*>(dAqk_ptr + gmem_addr);
                float4 kv = *reinterpret_cast<const float4*>(dAkk_ptr + gmem_addr);
                sDAqk(c + 0, r) = (c + 0 <= r) ? qv.x : 0.f;
                sDAqk(c + 1, r) = (c + 1 <= r) ? qv.y : 0.f;
                sDAqk(c + 2, r) = (c + 2 <= r) ? qv.z : 0.f;
                sDAqk(c + 3, r) = (c + 3 <= r) ? qv.w : 0.f;
                sDAkk(c + 0, r) = (c + 0 <= r) ? kv.x : 0.f;
                sDAkk(c + 1, r) = (c + 1 <= r) ? kv.y : 0.f;
                sDAkk(c + 2, r) = (c + 2 <= r) ? kv.z : 0.f;
                sDAkk(c + 3, r) = (c + 3 <= r) ? kv.w : 0.f;
            }
        } else {
            for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
                int r = idx / BC, c = idx % BC;  // r=j-row, c=i-col (contiguous in gmem)
                bool mask = (c <= r) && (i_ti + r < T_seq) && (i_ti + c < T_seq);
                int gmem_addr = (bos + i_ti + r) * (H * BT) + i_h * BT + i_i * BC + c;
                sDAqk(c, r) = mask ? dAqk_ptr[gmem_addr] : 0.f;
                sDAkk(c, r) = mask ? dAkk_ptr[gmem_addr] : 0.f;
            }
        }

        // Build Q_exp = q * exp2(g - gn), KB_exp = k * beta * exp2(g - gn) from persistent smem
        for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
            int n = idx % BK;
            int k_dim = idx / BK;
            bool valid = (i_ti + k_dim) < T_seq;
            float g_diff = valid ? (sG(k_dim, n) - smem.s_gn[n]) : 0.f;
            float exp_g = valid ? exp2f(g_diff) : 0.f;
            sKG(n, k_dim) = valid ? __bfloat162float(sQ(k_dim, n)) * exp_g : 0.f;
            float kv = valid ? __bfloat162float(sK(k_dim, n)) : 0.f;
            sKBG(n, k_dim) = kv * smem.s_beta[k_dim] * exp_g;
        }
        __syncthreads();

        float tmp_q[4] = {0.f, 0.f, 0.f, 0.f};
        float tmp_k[4] = {0.f, 0.f, 0.f, 0.f};
        gemm_m16n32k16(smem.s_dA_qk.data(), smem.s_KG.data(), tmp_q, tid);
        gemm_m16n32k16(smem.s_dA_kk.data(), smem.s_KBG.data(), tmp_k, tid);

        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            bool valid = (i_ti + row) < T_seq;
            float scale = valid ? exp2f(smem.s_gn[col] - sG(row, col)) : 0.f;
            dkt_acc[v] += (tmp_q[v] + tmp_k[v]) * scale;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // FINAL EPILOGUE: dk_out, dg_out (coalesced via smem staging)
    // ════════════════════════════════════════════════════════════════════════
    __syncthreads();
    {
        auto gDk_tile = local_tile(mDk, tile_hk, make_coord(tile_row, i_k));
        auto gDkOut_tile = local_tile(mDkOut, tile_hk, make_coord(tile_row, i_k));
        auto gDg_tile = local_tile(mDg, tile_hk, make_coord(tile_row, i_k));
        auto gDgOut_tile = local_tile(mDgOut, tile_hk, make_coord(tile_row, i_k));
        auto sStage = make_tensor(make_smem_ptr(smem.s_acc.data()), SmemLayoutAcc{});

        // ── dg_out: scatter compute to smem (vec float2 path) ──
        for (int rh = 0; rh < 2; ++rh) {
            int row, col_base;
            get_acc_pair_row_col(tid, rh, row, col_base);
            // qv, kv at (row, col_base) and (row, col_base+1) — bf16 packed
            uint32_t qpack = *reinterpret_cast<uint32_t*>(&sQ(row, col_base));
            uint32_t kpack = *reinterpret_cast<uint32_t*>(&sK(row, col_base));
            __nv_bfloat162 q2 = bitcast_bf162(qpack);
            __nv_bfloat162 k2 = bitcast_bf162(kpack);
            float2 qf = __bfloat1622float2(q2);
            float2 kf = __bfloat1622float2(k2);
            float2 prev{0.f, 0.f};
            if (!is_boundary || (i_ti + row) < T_seq) {
                prev = *reinterpret_cast<const float2*>(&gDg_tile(row, col_base));
            }
            float2 out;
            out.x = qf.x * dq2_acc[rh * 2 + 0] + (dk2_acc[rh * 2 + 0] - dkt_acc[rh * 2 + 0]) * kf.x + prev.x;
            out.y = qf.y * dq2_acc[rh * 2 + 1] + (dk2_acc[rh * 2 + 1] - dkt_acc[rh * 2 + 1]) * kf.y + prev.y;
            *reinterpret_cast<float2*>(&sStage(row, col_base)) = out;
        }
        __syncthreads();

        // BC*BK/4 = 128 = NUM_THREADS → 1 float4 store per thread
        {
            constexpr int VEC_ELEMS = BC * BK / 4;  // 128
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
        __syncthreads();

        // Preload old dk row-major so the hot dk_prev reads avoid accumulator-mapped gmem gathers.
        {
            constexpr int VEC_ELEMS = BC * BK / 4;  // 128
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
        }
        __syncthreads();

        // ── dk_out: scatter compute to smem, vectorized bf16x4 store (vec float2 path) ──
        for (int rh = 0; rh < 2; ++rh) {
            int row, col_base;
            get_acc_pair_row_col(tid, rh, row, col_base);
            float2* p = reinterpret_cast<float2*>(&sStage(row, col_base));
            float2 cur = *p;
            cur.x += dk2_acc[rh * 2 + 0] + dkt_acc[rh * 2 + 0];
            cur.y += dk2_acc[rh * 2 + 1] + dkt_acc[rh * 2 + 1];
            *p = cur;
        }
        __syncthreads();

        // BC*BK/8 = 64, 128 threads → 2 threads per chunk, half idle
        // Use 4-wide bf16 stores (64-bit): BC*BK/4 = 128 = NUM_THREADS
        {
            int vi = tid;
            int r = (vi * 4) / BK;
            int c = (vi * 4) % BK;
            if ((i_ti + r) < T_seq) {
                __nv_bfloat162 lo = __floats2bfloat162_rn(sStage(r, c + 0), sStage(r, c + 1));
                __nv_bfloat162 hi = __floats2bfloat162_rn(sStage(r, c + 2), sStage(r, c + 3));
                uint2 packed;
                packed.x = reinterpret_cast<uint32_t&>(lo);
                packed.y = reinterpret_cast<uint32_t&>(hi);
                *reinterpret_cast<uint2*>(&gDkOut_tile(r, c)) = packed;
            }
        }
    }
}

// ============================================================
// Host launch
// ============================================================
namespace sm90 {

void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream) {
    const int H = params.h;
    const int K = params.d;
    const int total = params.total_q_len;

    // All buffers now loaded via cp.async/cooperative in kernel (swizzled smem)
    // TMA descriptors in params struct are unused but kept for ABI compatibility

    // ── Launch kernel ──
    dim3 grid = NaiveTileScheduler::get_grid_shape(params.tile_scheduler_params);
    dim3 block(NUM_THREADS);
    int smem_size = sizeof(SmemStorage);

    kda_bwd_intra_kernel_sm90<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace sm90
