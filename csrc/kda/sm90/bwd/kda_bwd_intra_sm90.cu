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
#include "kda_bwd_utils.h"

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
constexpr int NUM_BUF = 2;                    // double buffer stages
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
    array_aligned<float, cosize_v<SmemLayoutDA>> smem_daqk[NUM_BUF];  // 16 KB
    array_aligned<float, cosize_v<SmemLayoutDA>> smem_dakk[NUM_BUF];  // 16 KB

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
    array_aligned<float, T_TILE> beta_smem[NUM_BUF];  // double-buffered with dA
    array_aligned<float, T_TILE> db_accum;   // 256 B (cross-ki dB accumulator)

    // ── Pipeline barriers ──
    // Load → consumers
    alignas(16) cute::uint64_t bar_load_qkg_ready[NUM_BUF];
    alignas(16) cute::uint64_t bar_load_dA_ready[NUM_BUF];

    // Prep → MMA
    alignas(16) cute::uint64_t bar_kg_all_ready, bar_qkg_all_ready;

    // Async pipelines: MMA → Prep
    alignas(16) cute::uint64_t bar_mma_ki_ready[NUM_BUF];  // dQ+dK per-ki output ready

    // Prep → Load
    alignas(16) cute::uint64_t bar_dA_free[NUM_BUF];
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
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_load_dA_ready[i], 2);  // TMA + elected Load thread
        }

        // Prep → MMA
        cute::initialize_barrier(smem->bar_kg_all_ready, 256);
        cute::initialize_barrier(smem->bar_qkg_all_ready, 256);

        // Async barriers: MMA → Prep (arrival count = 128 for MMA WG)
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_mma_ki_ready[i], 128);
        }

        // Prep → Load
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_dA_free[i], 256);
            cute::initialize_barrier(smem->bar_buf_free[i], 256);
        }

        cutlass::arch::fence_barrier_init();
    }

    __syncthreads();

    // ── Phase tracking ──
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
            int tile_counter = 0;
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int A_phase = (state_phase >> (buf_idx_A + NUM_BUF)) & 1;
                
                if (tile_counter >= NUM_BUF) {
                    cute::wait_barrier(smem->bar_dA_free[buf_idx_A], A_phase);
                }

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

                    // Wait when a value buffer slot wraps around and is reused.
                    if (k_idx >= NUM_BUF) {
                        cute::wait_barrier(smem->bar_buf_free[buf_idx_value], local_phase);
                    }

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
                buf_idx_A = (buf_idx_A + 1) % NUM_BUF;
                ++tile_counter;
            }
        }

        // =================================================================
        // MMA warpgroup — mma.sync 4-pass sub-chunk loop
        // =================================================================
    } else if (role == WGRole::MMA) {
        cutlass::arch::warpgroup_reg_alloc<REG_MMA>();

        // Thread mapping within MMA warpgroup (128 threads = 4 warps)
        const int warp_id = thread_idx / 32;      // 0..3, each warp handles N=8 cols
        const int lane_id = idx_in_warp;
        const int group_id = lane_id / 4;       // 0..7 (row within 16-row tile)
        const int lane_in_group = lane_id % 4;  // 0..3 (K-stride)

        // N-dimension column offset for this warp
        const int n_offset = warp_id * 8;

        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int tile_idx = get<2>(blk_coord);
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

            // Wait for dAqk/dAkk to be loaded (once per tile)
            cute::wait_barrier(smem->bar_dA_ready, phase_dA);

            for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                int cur_buf = k_idx % NUM_BUF;

                // Wait for B operands
                cute::wait_barrier(smem->bar_kg_ready[cur_buf], phase_kg[cur_buf]);
                cute::wait_barrier(smem->bar_qg_ready[cur_buf], phase_qg[cur_buf]);
                cute::wait_barrier(smem->bar_kbg_ready[cur_buf], phase_kbg[cur_buf]);

                // SMEM pointers for this ki
                const float* daqk = smem->smem_daqk.data();
                const float* dakk = smem->smem_dakk.data();
                const float* kg = smem->smem_kg[cur_buf].data();
                const float* qg = smem->smem_qg[cur_buf].data();
                const float* kbg = smem->smem_kbg[cur_buf].data();
                const float* g_smem = smem->smem_g[cur_buf].data();

                // ─────────────────────────────────────────────────────
                // CuTe SM80_16x8x8_F32TF32TF32F32_TN register mapping
                // (colex linearization of MMA_Traits layouts):
                //   A[M=16,K=8]: a0→[gid, lig], a1→[gid+8, lig],
                //                a2→[gid, lig+4], a3→[gid+8, lig+4]
                //   B[N=8,K=8]:  b0→[gid, lig], b1→[gid, lig+4]
                //     (B is [N,K] col-major; k=lig maps to KG row, n=gid maps to d-col)
                //   C[M=16,N=8]: c0→[gid, lig*2], c1→[gid, lig*2+1],
                //                c2→[gid+8, lig*2], c3→[gid+8, lig*2+1]
                // where lig = lane_in_group, gid = group_id
                // ─────────────────────────────────────────────────────

                // C output column offsets (two adjacent cols per thread)
                const int c_col0 = n_offset + lane_in_group * 2;
                const int c_col1 = c_col0 + 1;

                // ── Pass 1: dQ = tril(dAqk, 0) × KG ──
                for (int is = 0; is < 4; ++is) {
                    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                    const int c_row0 = is * 16 + group_id;
                    const int c_row1 = c_row0 + 8;

                    for (int js = 0; js <= is; ++js) {
                        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;

                        for (int k_inner = 0; k_inner < 2; ++k_inner) {
                            // A: a0→dAqk[gid, lig], a1→[gid+8, lig],
                            //    a2→[gid, lig+4], a3→[gid+8, lig+4]
                            int a_row0 = is * 16 + group_id;
                            int a_row1 = a_row0 + 8;
                            int a_k0 = js * 16 + k_inner * 8 + lane_in_group;
                            int a_k1 = a_k0 + 4;

                            float a0 = daqk[a_row0 * T_TILE + a_k0];
                            float a1 = daqk[a_row1 * T_TILE + a_k0];
                            float a2 = daqk[a_row0 * T_TILE + a_k1];
                            float a3 = daqk[a_row1 * T_TILE + a_k1];

                            // Causal mask: j <= i  (col > row → zero)
                            if (is == js) {
                                if (a_k0 > a_row0)
                                    a0 = 0.0f;
                                if (a_k0 > a_row1)
                                    a1 = 0.0f;
                                if (a_k1 > a_row0)
                                    a2 = 0.0f;
                                if (a_k1 > a_row1)
                                    a3 = 0.0f;
                            }

                            // B: b0→B[n=gid,k=lig] → KG[k_row, d_col]
                            //    k_row = js*16 + k_inner*8 + lig, d_col = n_offset + gid
                            int b_k0 = js * 16 + k_inner * 8 + lane_in_group;
                            float b0 = kg[b_k0 * K_TILE + n_offset + group_id];
                            float b1 = kg[(b_k0 + 4) * K_TILE + n_offset + group_id];

                            MMA_TF32::fma(
                                t0, t1, t2, t3, f2u(a0), f2u(a1), f2u(a2), f2u(a3), f2u(b0), f2u(b1), t0, t1, t2, t3);
                        }

                        // Scaling: exp2f(G[i,d] - G_norm_js[d])
                        // G_norm cancels with KG's exp2(G_norm - G[j]): net = exp2(G[i]-G[j])
                        float gn0 = g_smem[(js * 16) * K_TILE + c_col0];
                        float gn1 = g_smem[(js * 16) * K_TILE + c_col1];

                        c0 += t0 * safe_exp2f(g_smem[c_row0 * K_TILE + c_col0] - gn0);
                        c1 += t1 * safe_exp2f(g_smem[c_row0 * K_TILE + c_col1] - gn1);
                        c2 += t2 * safe_exp2f(g_smem[c_row1 * K_TILE + c_col0] - gn0);
                        c3 += t3 * safe_exp2f(g_smem[c_row1 * K_TILE + c_col1] - gn1);
                    }

                    // R2S: write dQ fragment to smem_dq_out
                    smem->smem_dq_out[cur_buf][c_row0 * K_TILE + c_col0] = c0;
                    smem->smem_dq_out[cur_buf][c_row0 * K_TILE + c_col1] = c1;
                    smem->smem_dq_out[cur_buf][c_row1 * K_TILE + c_col0] = c2;
                    smem->smem_dq_out[cur_buf][c_row1 * K_TILE + c_col1] = c3;
                }

                // ── Pass 2: dK_lower = tril(dAkk, 0) × KG (including diagonal) ──
                float dk_reg[4 * 4];
                for (int i = 0; i < 16; ++i)
                    dk_reg[i] = 0.0f;

                for (int is = 0; is < 4; ++is) {
                    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

                    for (int js = 0; js <= is; ++js) {
                        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;

                        for (int k_inner = 0; k_inner < 2; ++k_inner) {
                            int a_row0 = is * 16 + group_id;
                            int a_row1 = a_row0 + 8;
                            int a_k0 = js * 16 + k_inner * 8 + lane_in_group;
                            int a_k1 = a_k0 + 4;

                            float a0 = dakk[a_row0 * T_TILE + a_k0];
                            float a1 = dakk[a_row1 * T_TILE + a_k0];
                            float a2 = dakk[a_row0 * T_TILE + a_k1];
                            float a3 = dakk[a_row1 * T_TILE + a_k1];

                            // Causal mask on diagonal: i >= j (col <= row)
                            if (is == js) {
                                if (a_k0 > a_row0)
                                    a0 = 0.0f;
                                if (a_k0 > a_row1)
                                    a1 = 0.0f;
                                if (a_k1 > a_row0)
                                    a2 = 0.0f;
                                if (a_k1 > a_row1)
                                    a3 = 0.0f;
                            }

                            int b_k0 = js * 16 + k_inner * 8 + lane_in_group;
                            float b0 = kg[b_k0 * K_TILE + n_offset + group_id];
                            float b1 = kg[(b_k0 + 4) * K_TILE + n_offset + group_id];

                            MMA_TF32::fma(
                                t0, t1, t2, t3, f2u(a0), f2u(a1), f2u(a2), f2u(a3), f2u(b0), f2u(b1), t0, t1, t2, t3);
                        }

                        int cr0 = is * 16 + group_id;
                        int cr1 = cr0 + 8;
                        float gn0 = g_smem[(js * 16) * K_TILE + c_col0];
                        float gn1 = g_smem[(js * 16) * K_TILE + c_col1];

                        c0 += t0 * safe_exp2f(g_smem[cr0 * K_TILE + c_col0] - gn0);
                        c1 += t1 * safe_exp2f(g_smem[cr0 * K_TILE + c_col1] - gn1);
                        c2 += t2 * safe_exp2f(g_smem[cr1 * K_TILE + c_col0] - gn0);
                        c3 += t3 * safe_exp2f(g_smem[cr1 * K_TILE + c_col1] - gn1);
                    }

                    dk_reg[is * 4 + 0] = c0;
                    dk_reg[is * 4 + 1] = c1;
                    dk_reg[is * 4 + 2] = c2;
                    dk_reg[is * 4 + 3] = c3;
                }

                // ── Pass 3+4: dK_upper = triu(dAqk)^T × QG + strict_triu(dAkk)^T × KBG ──
                float dkt_reg[4 * 4];
                for (int i = 0; i < 16; ++i)
                    dkt_reg[i] = 0.0f;

                for (int js = 0; js < 4; ++js) {
                    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

                    // Pass 3: dAqk^T × QG (is >= js)
                    for (int is = js; is < 4; ++is) {
                        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;

                        for (int k_inner = 0; k_inner < 2; ++k_inner) {
                            // Transposed A: A_T[m, k] = dAqk[k, m]
                            // a0→A_T[gid,lig] = dAqk[is*16+k8+lig, js*16+gid]
                            // a1→A_T[gid+8,lig] = dAqk[is*16+k8+lig, js*16+gid+8]
                            // a2→A_T[gid,lig+4] = dAqk[is*16+k8+lig+4, js*16+gid]
                            // a3→A_T[gid+8,lig+4] = dAqk[is*16+k8+lig+4, js*16+gid+8]
                            int daqk_k0 = is * 16 + k_inner * 8 + lane_in_group;
                            int daqk_k1 = daqk_k0 + 4;
                            int m0 = js * 16 + group_id;
                            int m1 = m0 + 8;

                            float a0 = daqk[daqk_k0 * T_TILE + m0];
                            float a1 = daqk[daqk_k0 * T_TILE + m1];
                            float a2 = daqk[daqk_k1 * T_TILE + m0];
                            float a3 = daqk[daqk_k1 * T_TILE + m1];

                            // Causal mask: keep elements where i >= j
                            // i = daqk_k{0,1} (original row), j = m{0,1} (original col)
                            // zero when i < j (i.e., k < m in transposed view)
                            if (is == js) {
                                if (daqk_k0 < m0)
                                    a0 = 0.0f;
                                if (daqk_k0 < m1)
                                    a1 = 0.0f;
                                if (daqk_k1 < m0)
                                    a2 = 0.0f;
                                if (daqk_k1 < m1)
                                    a3 = 0.0f;
                            }

                            // B: QG, b0→B[n=gid,k=lig] → QG[k_row, d_col]
                            int b_k0 = is * 16 + k_inner * 8 + lane_in_group;
                            float b0 = qg[b_k0 * K_TILE + n_offset + group_id];
                            float b1 = qg[(b_k0 + 4) * K_TILE + n_offset + group_id];

                            MMA_TF32::fma(
                                t0, t1, t2, t3, f2u(a0), f2u(a1), f2u(a2), f2u(a3), f2u(b0), f2u(b1), t0, t1, t2, t3);
                        }

                        // K-side scaling: exp2f(G_norm_is - G[j])
                        int cr0 = js * 16 + group_id;
                        int cr1 = cr0 + 8;
                        float gn0 = g_smem[(is * 16) * K_TILE + c_col0];
                        float gn1 = g_smem[(is * 16) * K_TILE + c_col1];

                        t0 *= safe_exp2f(gn0 - g_smem[cr0 * K_TILE + c_col0]);
                        t1 *= safe_exp2f(gn1 - g_smem[cr0 * K_TILE + c_col1]);
                        t2 *= safe_exp2f(gn0 - g_smem[cr1 * K_TILE + c_col0]);
                        t3 *= safe_exp2f(gn1 - g_smem[cr1 * K_TILE + c_col1]);

                        c0 += t0;
                        c1 += t1;
                        c2 += t2;
                        c3 += t3;
                    }

                    // Pass 4: dAkk^T × KBG (is >= js, including diagonal)
                    for (int is = js; is < 4; ++is) {
                        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;

                        for (int k_inner = 0; k_inner < 2; ++k_inner) {
                            int dakk_k0 = is * 16 + k_inner * 8 + lane_in_group;
                            int dakk_k1 = dakk_k0 + 4;
                            int m0 = js * 16 + group_id;
                            int m1 = m0 + 8;

                            float a0 = dakk[dakk_k0 * T_TILE + m0];
                            float a1 = dakk[dakk_k0 * T_TILE + m1];
                            float a2 = dakk[dakk_k1 * T_TILE + m0];
                            float a3 = dakk[dakk_k1 * T_TILE + m1];

                            // Causal mask on diagonal: keep k >= m (j >= i)
                            if (is == js) {
                                if (dakk_k0 < m0)
                                    a0 = 0.0f;
                                if (dakk_k0 < m1)
                                    a1 = 0.0f;
                                if (dakk_k1 < m0)
                                    a2 = 0.0f;
                                if (dakk_k1 < m1)
                                    a3 = 0.0f;
                            }

                            int b_k0 = is * 16 + k_inner * 8 + lane_in_group;
                            float b0 = kbg[b_k0 * K_TILE + n_offset + group_id];
                            float b1 = kbg[(b_k0 + 4) * K_TILE + n_offset + group_id];

                            MMA_TF32::fma(
                                t0, t1, t2, t3, f2u(a0), f2u(a1), f2u(a2), f2u(a3), f2u(b0), f2u(b1), t0, t1, t2, t3);
                        }

                        int cr0 = js * 16 + group_id;
                        int cr1 = cr0 + 8;
                        float gn0 = g_smem[(is * 16) * K_TILE + c_col0];
                        float gn1 = g_smem[(is * 16) * K_TILE + c_col1];

                        t0 *= safe_exp2f(gn0 - g_smem[cr0 * K_TILE + c_col0]);
                        t1 *= safe_exp2f(gn1 - g_smem[cr0 * K_TILE + c_col1]);
                        t2 *= safe_exp2f(gn0 - g_smem[cr1 * K_TILE + c_col0]);
                        t3 *= safe_exp2f(gn1 - g_smem[cr1 * K_TILE + c_col1]);

                        c0 += t0;
                        c1 += t1;
                        c2 += t2;
                        c3 += t3;
                    }

                    dkt_reg[js * 4 + 0] = c0;
                    dkt_reg[js * 4 + 1] = c1;
                    dkt_reg[js * 4 + 2] = c2;
                    dkt_reg[js * 4 + 3] = c3;
                }

                // ── R2S: dk_lower → smem_dk_out, dk_upper → smem_dk_upper_out ──
                {
                    for (int s = 0; s < 4; ++s) {
                        int r0 = s * 16 + group_id;
                        int r1 = r0 + 8;
                        // dK_lower (Pass 2)
                        smem->smem_dk_out[cur_buf][r0 * K_TILE + c_col0] = dk_reg[s * 4 + 0];
                        smem->smem_dk_out[cur_buf][r0 * K_TILE + c_col1] = dk_reg[s * 4 + 1];
                        smem->smem_dk_out[cur_buf][r1 * K_TILE + c_col0] = dk_reg[s * 4 + 2];
                        smem->smem_dk_out[cur_buf][r1 * K_TILE + c_col1] = dk_reg[s * 4 + 3];
                        // dK_upper (Pass 3+4)
                        smem->smem_dk_upper_out[cur_buf][r0 * K_TILE + c_col0] = dkt_reg[s * 4 + 0];
                        smem->smem_dk_upper_out[cur_buf][r0 * K_TILE + c_col1] = dkt_reg[s * 4 + 1];
                        smem->smem_dk_upper_out[cur_buf][r1 * K_TILE + c_col0] = dkt_reg[s * 4 + 2];
                        smem->smem_dk_upper_out[cur_buf][r1 * K_TILE + c_col1] = dkt_reg[s * 4 + 3];
                    }
                }

                fence_view_async_shared();

                // Signal Prep: dQ+dK for this ki ready
                cute::arrive_barrier(smem->bar_mma_ki_ready[cur_buf]);

                phase_kg[cur_buf] ^= 1;
                phase_qg[cur_buf] ^= 1;
                phase_kbg[cur_buf] ^= 1;
            }

            phase_dA ^= 1;
        }

        // =================================================================
        // Prep warpgroups — scalar computation
        // =================================================================
    } else {
        cutlass::arch::warpgroup_reg_alloc<REG_PREP>();

        // Prep thread ID within the 256-thread Prep group
        int prep_tid = thread_idx - PREP_THREAD_OFFSET;  // [0, 255]

        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            // ── Wait for dAqk/dAkk TMA load ──
            cute::wait_barrier(smem->bar_dA_ready, phase_dA);

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int token_offset = cu_len_ptr[batch_idx];
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

            // Initialize dB accumulator
            if (prep_tid < T_TILE) {
                smem->db_accum[prep_tid] = 0.0f;
            }
            fence_view_async_shared();

            // ── Per-ki loop ──
            for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                int cur_buf = k_idx % NUM_BUF;

                // Wait for Q, K, G data
                cute::wait_barrier(smem->bar_qkg_ready[cur_buf], phase_qkg[cur_buf]);

                // Zero G at padding rows to prevent NaN in exp2f gate scaling
                // (both B operand construction and MMA gate scaling read G from SMEM)
                if (sub_seq_len < T_TILE) {
                    float* g_ptr = smem->smem_g[cur_buf].data();
                    for (int elem = prep_tid; elem < (T_TILE - sub_seq_len) * K_TILE; elem += 256) {
                        g_ptr[(sub_seq_len + elem / K_TILE) * K_TILE + elem % K_TILE] = 0.0f;
                    }
                    fence_view_async_shared();
                }

                // ── Construct KG: KG[j,d] = K[j,d] * exp2f(G_norm[d] - G[j,d]) ──
                // Per-subchunk G_norm: 4 sub-tiles, each 16 rows
                // Zero padding rows (j >= sub_seq_len) to avoid NaN from cross-sequence gate values
                {
                    float* kg_ptr = smem->smem_kg[cur_buf].data();
                    float* g_ptr = smem->smem_g[cur_buf].data();
                    bf16* k_ptr = reinterpret_cast<bf16*>(smem->smem_k[cur_buf].data());

                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int j = elem / K_TILE;
                        int d = elem % K_TILE;

                        if (j < sub_seq_len) {
                            int js = j / SUB_T_TILE;           // sub-tile index
                            int g_norm_row = js * SUB_T_TILE;  // G_norm = G[js*16, d]

                            float g_val = g_ptr[j * K_TILE + d];
                            float g_norm = g_ptr[g_norm_row * K_TILE + d];
                            float k_val = static_cast<float>(k_ptr[j * K_TILE + d]);

                            kg_ptr[j * K_TILE + d] = k_val * safe_exp2f(g_norm - g_val);
                        } else {
                            kg_ptr[j * K_TILE + d] = 0.0f;
                        }
                    }
                }

                // ── Construct QG: QG[i,d] = Q[i,d] * exp2f(G[i,d] - G_norm[d]) ──
                {
                    float* qg_ptr = smem->smem_qg[cur_buf].data();
                    float* g_ptr = smem->smem_g[cur_buf].data();
                    bf16* q_ptr = reinterpret_cast<bf16*>(smem->smem_q[cur_buf].data());

                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int i = elem / K_TILE;
                        int d = elem % K_TILE;

                        if (i < sub_seq_len) {
                            int is = i / SUB_T_TILE;
                            int g_norm_row = is * SUB_T_TILE;

                            float g_val = g_ptr[i * K_TILE + d];
                            float g_norm = g_ptr[g_norm_row * K_TILE + d];
                            float q_val = static_cast<float>(q_ptr[i * K_TILE + d]);

                            qg_ptr[i * K_TILE + d] = q_val * safe_exp2f(g_val - g_norm);
                        } else {
                            qg_ptr[i * K_TILE + d] = 0.0f;
                        }
                    }
                }

                // ── Construct KBG: KBG[i,d] = K[i,d] * beta[i] * exp2f(G[i,d] - G_norm[d]) ──
                {
                    float* kbg_ptr = smem->smem_kbg[cur_buf].data();
                    float* g_ptr = smem->smem_g[cur_buf].data();
                    bf16* k_ptr = reinterpret_cast<bf16*>(smem->smem_k[cur_buf].data());

                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int i = elem / K_TILE;
                        int d = elem % K_TILE;

                        if (i < sub_seq_len) {
                            int is = i / SUB_T_TILE;
                            int g_norm_row = is * SUB_T_TILE;

                            float g_val = g_ptr[i * K_TILE + d];
                            float g_norm = g_ptr[g_norm_row * K_TILE + d];
                            float k_val = static_cast<float>(k_ptr[i * K_TILE + d]);
                            float beta_val = smem->beta_smem[i];

                            kbg_ptr[i * K_TILE + d] = k_val * beta_val * safe_exp2f(g_val - g_norm);
                        } else {
                            kbg_ptr[i * K_TILE + d] = 0.0f;
                        }
                    }
                }

                fence_view_async_shared();

                // ── Debug output: write B operands to HBM (when enabled) ──
                if (params.debug_kg_ptr || params.debug_qg_ptr || params.debug_kbg_ptr) {
                    int stride_hd = params.h * K_SIZE;
                    int base_offset =
                        (token_offset + tile_idx * T_TILE) * params.h * K_SIZE + head_idx * K_SIZE + k_idx * K_TILE;
                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int row = elem / K_TILE;
                        int col = elem % K_TILE;
                        if (row < sub_seq_len) {
                            int out_idx = base_offset + row * stride_hd + col;
                            if (params.debug_kg_ptr)
                                ((float*)params.debug_kg_ptr)[out_idx] = smem->smem_kg[cur_buf][elem];
                            if (params.debug_qg_ptr)
                                ((float*)params.debug_qg_ptr)[out_idx] = smem->smem_qg[cur_buf][elem];
                            if (params.debug_kbg_ptr)
                                ((float*)params.debug_kbg_ptr)[out_idx] = smem->smem_kbg[cur_buf][elem];
                        }
                    }
                }

                // ── Signal MMA: B operands ready ──
                cute::arrive_barrier(smem->bar_kg_ready[cur_buf]);
                cute::arrive_barrier(smem->bar_qg_ready[cur_buf]);
                cute::arrive_barrier(smem->bar_kbg_ready[cur_buf]);

                // ── Wait for MMA output ──
                cute::wait_barrier(smem->bar_mma_ki_ready[cur_buf], phase_mma_ki[cur_buf]);

                // ── Epilogue per ki: dB accumulation, beta scaling, dQ/dK/dG output ──
                {
                    float* dq_out_base = (float*)params.dq_out_ptr +
                                         (token_offset + tile_idx * T_TILE) * params.h * K_SIZE + head_idx * K_SIZE +
                                         k_idx * K_TILE;
                    float* dk_out_base = (float*)params.dk_out_ptr +
                                         (token_offset + tile_idx * T_TILE) * params.h * K_SIZE + head_idx * K_SIZE +
                                         k_idx * K_TILE;
                    float* dg_out_base = (float*)params.dg_out_ptr +
                                         (token_offset + tile_idx * T_TILE) * params.h * K_SIZE + head_idx * K_SIZE +
                                         k_idx * K_TILE;

                    int stride_hd = params.h * K_SIZE;

                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int row = elem / K_TILE;
                        int col = elem % K_TILE;
                        if (row < sub_seq_len) {
                            // Read MMA outputs (separate dk_lower and dk_upper)
                            float dq_intra = smem->smem_dq_out[cur_buf][row * K_TILE + col];
                            float dk_lower = smem->smem_dk_out[cur_buf][row * K_TILE + col];
                            float dk_upper = smem->smem_dk_upper_out[cur_buf][row * K_TILE + col];

                            // Read inter-chunk inputs
                            float dq_inter = smem->smem_dq_in[cur_buf][row * K_TILE + col];
                            float dk_inter = smem->smem_dk_in[cur_buf][row * K_TILE + col];
                            float dg_inter = smem->smem_dg_in[cur_buf][row * K_TILE + col];

                            // Read Q, K for dB/dG computation
                            float q_val = static_cast<float>(
                                reinterpret_cast<bf16*>(smem->smem_q[cur_buf].data())[row * K_TILE + col]);
                            float k_val = static_cast<float>(
                                reinterpret_cast<bf16*>(smem->smem_k[cur_buf].data())[row * K_TILE + col]);
                            float beta_i = smem->beta_smem[row];

                            // dB: accumulate dk_lower * K BEFORE beta scaling
                            atomicAdd(&smem->db_accum[row], dk_lower * k_val);

                            // Beta scaling on dK_lower
                            float dk_lower_beta = dk_lower * beta_i;

                            // dQ_final = dQ_inter + dQ_intra
                            dq_out_base[row * stride_hd + col] = dq_inter + dq_intra;

                            // dK_final = dK_inter + dK_lower*beta + dK_upper
                            dk_out_base[row * stride_hd + col] = dk_inter + dk_lower_beta + dk_upper;

                            // dG = dG_inter + Q*dQ_intra + (dK_lower*beta - dK_upper)*K
                            dg_out_base[row * stride_hd + col] =
                                dg_inter + q_val * dq_intra + (dk_lower_beta - dk_upper) * k_val;
                        }
                    }
                }

                // Signal buffer free for next ki TMA load
                cute::arrive_barrier(smem->bar_buf_free[cur_buf]);

                phase_qkg[cur_buf] ^= 1;
                phase_mma_ki[cur_buf] ^= 1;
            }  // end per-ki loop

            // ── dB output: db_accum (cross-ki sum) + db_inter ──
            if (prep_tid < T_TILE && prep_tid < sub_seq_len) {
                int global_row = token_offset + tile_idx * T_TILE + prep_tid;
                float db_inter = ((float*)params.db_ptr)[global_row * params.h + head_idx];
                float* db_out = (float*)params.db_out_ptr + global_row * params.h + head_idx;
                *db_out = smem->db_accum[prep_tid] + db_inter;
            }

            phase_dA ^= 1;

            // Signal epilogue done (for LdSt warp3 dB TMA store, Phase 4)
            cute::arrive_barrier(smem->bar_epilogue_done);
        }  // end persistent tile loop
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
