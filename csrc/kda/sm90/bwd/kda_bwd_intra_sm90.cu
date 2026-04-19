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
// [32,16] fp32, row_bytes=64B → Swizzle<2,2,3> (TMA 64B mode)
using SmemLayoutB_op =
    decltype(composition(Swizzle<2, 2, 3>{}, Layout<Shape<Int<BK>, Int<BC>>, Stride<Int<BC>, _1>>{}));
// [16,32] fp32, row_bytes=128B → Swizzle<3,2,3> (128B mode)
using SmemLayoutAcc = decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));

// S2R atom for MMA operand loads
using S2RAtom = Copy_Atom<UniversalCopy<float>, float>;

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

    // S2R copies
    auto s2r_copy_a = make_tiled_copy_A(S2RAtom{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RAtom{}, tiled_mma);
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

    // ── Load persistent tiles via cp.async: Q, K, G + cooperative beta ──
    {
        // Q: bf16 [16,32] = 512 elems, 16B/chunk = 8 bf16 → 64 chunks
        {
            auto gQ_tile = local_tile(mQ, tile_hk, make_coord(tile_row, i_k));
            constexpr int Q_CHUNKS = (BC * BK) / 8;  // 64
            for (int ci = tid; ci < Q_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 8;
                int r = elem / BK;
                int c = elem % BK;
                uint32_t dst = cute::cast_smem_ptr_to_uint(&sQ(r, c));
                const __nv_bfloat16* src = &gQ_tile(r, c);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
            }
        }
        // K: bf16 [16,32] = 512 elems → 64 chunks
        {
            auto gK_tile = local_tile(mK, tile_hk, make_coord(tile_row, i_k));
            constexpr int K_CHUNKS = (BC * BK) / 8;  // 64
            for (int ci = tid; ci < K_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 8;
                int r = elem / BK;
                int c = elem % BK;
                uint32_t dst = cute::cast_smem_ptr_to_uint(&sK(r, c));
                const __nv_bfloat16* src = &gK_tile(r, c);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
            }
        }
        // G: fp32 [16,32] = 512 elems, 16B/chunk = 4 fp32 → 128 chunks
        {
            auto gG_tile = local_tile(mG, tile_hk, make_coord(tile_row, i_k));
            constexpr int G_CHUNKS = (BC * BK) / 4;  // 128
            for (int ci = tid; ci < G_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 4;
                int r = elem / BK;
                int c = elem % BK;
                uint32_t dst = cute::cast_smem_ptr_to_uint(&sG(r, c));
                const float* src = &gG_tile(r, c);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
            }
        }
        // Beta: cooperative (scalar per row, not worth cp.async)
        {
            auto gBeta_tile = local_tile(mBeta, Int<BC>{}, tile_row);
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

        for (int i_j = 0; i_j < i_i; ++i_j) {
            int j_tile = i_t * NC + i_j;

            // cp.async load dAqk + dAkk: fp32 [16,16] = 256 elems, 4 per chunk → 64 chunks
            {
                auto gDAqk_j = local_tile(mDAqk_g, tile_da, make_coord(tile_row, i_j));
                auto gDAkk_j = local_tile(mDAkk_g, tile_da, make_coord(tile_row, i_j));
                constexpr int DA_CHUNKS = (BC * BC) / 4;  // 64
                for (int ci = tid; ci < DA_CHUNKS; ci += NUM_THREADS) {
                    int elem = ci * 4;
                    int r = elem / BC;
                    int c = elem % BC;
                    uint32_t dst_qk = cute::cast_smem_ptr_to_uint(&sDAqk(r, c));
                    uint32_t dst_kk = cute::cast_smem_ptr_to_uint(&sDAkk(r, c));
                    const float* src_qk = &gDAqk_j(r, c);
                    const float* src_kk = &gDAkk_j(r, c);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_qk), "l"(src_qk));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_kk), "l"(src_kk));
                }
            }

            // Build KG operand from gmem (overlaps TMA)
            {
                auto gK_j = local_tile(mK, tile_hk, make_coord(j_tile, i_k));
                auto gG_j = local_tile(mG, tile_hk, make_coord(j_tile, i_k));
                for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
                    int n = idx % BK;
                    int k_dim = idx / BK;
                    float kv = __bfloat162float(gK_j(k_dim, n));
                    float gv = gG_j(k_dim, n);
                    sKG(n, k_dim) = kv * exp2f(smem.s_gn[n] - gv);
                }
            }

            // Wait for cp.async dA + fence KG writes
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();

            gemm_m16n32k16(smem.s_dA_qk.data(), smem.s_KG.data(), dq2_acc, tid);
            gemm_m16n32k16(smem.s_dA_kk.data(), smem.s_KG.data(), dk2_acc, tid);
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
            for (int ci = tid; ci < DA_CHUNKS; ci += NUM_THREADS) {
                int elem = ci * 4;
                int r = elem / BC;
                int c = elem % BC;
                uint32_t dst_qk = cute::cast_smem_ptr_to_uint(&sDAqk(r, c));
                uint32_t dst_kk = cute::cast_smem_ptr_to_uint(&sDAkk(r, c));
                const float* src_qk = &gDAqk_diag(r, c);
                const float* src_kk = &gDAkk_diag(r, c);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_qk), "l"(src_qk));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_kk), "l"(src_kk));
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
        gemm_m16n32k16(smem.s_dA_qk.data(), smem.s_KG.data(), tmp_dq, tid);
        gemm_m16n32k16(smem.s_dA_kk.data(), smem.s_KG.data(), tmp_dk, tid);

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

        // Scatter: each thread computes final dq value and writes to smem
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            sStage(row, col) = dq2_acc[v] + gDq_tile(row, col);
        }
        __syncthreads();

        // Coalesced store to gmem as bf16
        for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            if ((i_ti + r) < T_seq) {
                gDqOut_tile(r, c) = __float2bfloat16(sStage(r, c));
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

        for (int i_j = i_i + 1; i_j < NC_eff; ++i_j) {
            int j_tile = i_t * NC + i_j;
            int j_ti = i_t * BT + i_j * BC;

            // Cooperative transposed loads: sDA(r,c) = gDA[j_row+c, i_col+r]
            for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
                int r = idx / BC, c = idx % BC;
                bool valid = (j_ti + c) < T_seq;
                int gmem_addr = (bos + j_ti + c) * (H * BT) + i_h * BT + i_i * BC + r;
                sDAqk(r, c) = valid ? dAqk_ptr[gmem_addr] : 0.f;
                sDAkk(r, c) = valid ? dAkk_ptr[gmem_addr] : 0.f;
            }

            // Build QG = q_j * exp2(g_j - gn) and KBG = k_j * beta_j * exp2(g_j - gn)
            {
                auto gQ_j = local_tile(mQ, tile_hk, make_coord(j_tile, i_k));
                auto gK_j = local_tile(mK, tile_hk, make_coord(j_tile, i_k));
                auto gG_j = local_tile(mG, tile_hk, make_coord(j_tile, i_k));
                auto gBeta_j = local_tile(mBeta, Int<BC>{}, j_tile);
                for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
                    int n = idx % BK;
                    int k_dim = idx / BK;
                    bool valid = (j_ti + k_dim) < T_seq;
                    float gv = valid ? gG_j(k_dim, n) : 0.f;
                    float gating = valid ? exp2f(gv - smem.s_gn[n]) : 0.f;
                    float qv = valid ? __bfloat162float(gQ_j(k_dim, n)) : 0.f;
                    sKG(n, k_dim) = qv * gating;
                    float kv = valid ? __bfloat162float(gK_j(k_dim, n)) : 0.f;
                    float bv = valid ? __bfloat162float(gBeta_j(k_dim)) : 0.f;
                    sKBG(n, k_dim) = kv * bv * gating;
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
            dkt_acc[v] *= exp2f(smem.s_gn[col] - sG(row, col));
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

        // Cooperative transposed loads with upper-tri mask (r <= c)
        for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
            int r = idx / BC, c = idx % BC;
            bool mask = (r <= c) && (i_ti + r < T_seq) && (i_ti + c < T_seq);
            int gmem_addr = (bos + i_ti + c) * (H * BT) + i_h * BT + i_i * BC + r;
            sDAqk(r, c) = mask ? dAqk_ptr[gmem_addr] : 0.f;
            sDAkk(r, c) = mask ? dAkk_ptr[gmem_addr] : 0.f;
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

        // ── dg_out: scatter compute to smem, coalesced store fp32 ──
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            float qv = __bfloat162float(sQ(row, col));
            float kv = __bfloat162float(sK(row, col));
            sStage(row, col) = qv * dq2_acc[v] + (dk2_acc[v] - dkt_acc[v]) * kv + gDg_tile(row, col);
        }
        __syncthreads();

        for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            if ((i_ti + r) < T_seq) {
                gDgOut_tile(r, c) = sStage(r, c);
            }
        }
        __syncthreads();

        // ── dk_out: scatter compute to smem, coalesced store bf16 ──
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            sStage(row, col) = dk2_acc[v] + gDk_tile(row, col) + dkt_acc[v];
        }
        __syncthreads();

        for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            if ((i_ti + r) < T_seq) {
                gDkOut_tile(r, c) = __float2bfloat16(sStage(r, c));
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
