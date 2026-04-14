// SM90 KDA Backward Intra-Chunk Kernel — Phase 1: Framework + Infrastructure
//
// Design: 512 threads = 4 WG (LdSt / MMA / Prep×2)
// MMA: SM80 mma.sync TF32 sub-chunk loop (16×8×8)
// B operands: fp32 row-major (hardware truncation to TF32)
// Causal mask: on-the-fly in registers during A operand loading

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
constexpr int NUM_THREADS = 512;              // 4 WG × 128
constexpr int NUM_WG = 4;
constexpr int CHUNK_SIZE = 64;

// Register allocation
constexpr int REG_LDST = 24;   // LdSt: minimal registers
constexpr int REG_MMA = 168;   // MMA: fragment-heavy
constexpr int REG_PREP = 160;  // Prep: scalar computation

// =====================================================================
// WG Role Assignment
// =====================================================================
// WG0 (threads 0-127):   LdSt
// WG1 (threads 128-255): MMA
// WG2 (threads 256-383): Prep
// WG3 (threads 384-511): Prep

enum class WGRole {
    LdSt = 0,
    MMA = 1,
    Prep = 2,
};

__forceinline__ __device__ WGRole
get_wg_role() {
    int wg_idx = threadIdx.x / 128;
    if (wg_idx == 0)
        return WGRole::LdSt;
    if (wg_idx == 1)
        return WGRole::MMA;
    return WGRole::Prep;  // WG2, WG3
}

__forceinline__ __device__ int
get_warp_idx_in_wg() {
    return (threadIdx.x % 128) / 32;
}

// =====================================================================
// SMEM Layouts
// =====================================================================

// Q, K: bf16 [64, 32] — TMA loaded
using SmemLayoutInputBF16 = Layout<Shape<Int<T_TILE>, Int<K_TILE>>, Stride<Int<K_TILE>, _1>>;

// G, dQ_in, dK_in, dG_in, dQ_out, dK_out: fp32 [64, 32] — TMA loaded / MMA R2S
using SmemLayoutInputFP32 = Layout<Shape<Int<T_TILE>, Int<K_TILE>>, Stride<Int<K_TILE>, _1>>;

// dAqk, dAkk: fp32 [64, 64] — TMA loaded
using SmemLayoutDA = Layout<Shape<Int<T_TILE>, Int<T_TILE>>, Stride<Int<T_TILE>, _1>>;

// KG, QG, KBG: fp32 [64, 32] — Prep element-wise, MMA thread load
using SmemLayoutB = Layout<Shape<Int<T_TILE>, Int<K_TILE>>, Stride<Int<K_TILE>, _1>>;

// Beta: fp32 [64]
using SmemLayoutBeta = Layout<Shape<Int<T_TILE>>, Stride<_1>>;

// dB: fp32 [64]
using SmemLayoutDB = Layout<Shape<Int<T_TILE>>, Stride<_1>>;

// =====================================================================
// SharedStorage (~201 KB, within 228 KB limit)
// =====================================================================
struct SharedStorage {
    // TMA-loaded input buffers (double-buffered per ki)
    alignas(128) cute::array_aligned<bf16, T_TILE * K_TILE> smem_q[NUM_BUF];   // 2 × 4 KB = 8 KB
    alignas(128) cute::array_aligned<bf16, T_TILE * K_TILE> smem_k[NUM_BUF];   // 2 × 4 KB = 8 KB
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_g[NUM_BUF];  // 2 × 8 KB = 16 KB

    // dQ, dK, dG inter-chunk input (double-buffered per ki)
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_dq_in[NUM_BUF];  // 2 × 8 KB = 16 KB
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_dk_in[NUM_BUF];  // 2 × 8 KB = 16 KB
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_dg_in[NUM_BUF];  // 2 × 8 KB = 16 KB

    // dAqk, dAkk: loaded once per tile (single-buffered)
    alignas(128) cute::array_aligned<float, T_TILE * T_TILE> smem_daqk;  // 16 KB
    alignas(128) cute::array_aligned<float, T_TILE * T_TILE> smem_dakk;  // 16 KB

    // B operands for MMA: KG, QG, KBG (double-buffered per ki)
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_kg[NUM_BUF];   // 2 × 8 KB = 16 KB
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_qg[NUM_BUF];   // 2 × 8 KB = 16 KB
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_kbg[NUM_BUF];  // 2 × 8 KB = 16 KB

    // MMA output (double-buffered per ki)
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_dq_out[NUM_BUF];  // 2 × 8 KB = 16 KB
    alignas(128) cute::array_aligned<float, T_TILE * K_TILE> smem_dk_out[NUM_BUF];  // 2 × 8 KB = 16 KB

    // Scalar data
    alignas(16) float beta_smem[T_TILE];  // 256 B
    alignas(16) float db_accum[T_TILE];   // 256 B (cross-ki dB accumulator)

    // ── Pipeline barriers ──
    // TmaAsync pipelines: LdSt → Prep
    alignas(16) cute::uint64_t bar_qkg_ready[NUM_BUF];   // Q, K, G + dQ, dK, dG inter loaded
    alignas(16) cute::uint64_t bar_dA_ready;             // dAqk, dAkk, beta loaded

    // Async pipelines: Prep → MMA
    alignas(16) cute::uint64_t bar_mask_ready;          // raw dA data ready (Prep notifies MMA)
    alignas(16) cute::uint64_t bar_kg_ready[NUM_BUF];   // KG double-buffered
    alignas(16) cute::uint64_t bar_qg_ready[NUM_BUF];   // QG double-buffered
    alignas(16) cute::uint64_t bar_kbg_ready[NUM_BUF];  // KBG double-buffered

    // Async pipelines: MMA → Prep
    alignas(16) cute::uint64_t bar_mma_ki_ready[NUM_BUF];  // dQ+dK per-ki output ready

    // Async pipeline: Prep → LdSt
    alignas(16) cute::uint64_t bar_epilogue_done;  // dB ready for store

    // Flow control
    alignas(16) cute::uint64_t bar_buf_free[NUM_BUF];  // buffer free signal
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
    typename TMA_DG,
    typename TMA_BETA>
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
    TMA_BETA tma_beta;
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
    const int wg_idx = thread_idx / 128;
    const int idx_in_wg = thread_idx % 128;
    const int warp_idx_in_wg = idx_in_wg / 32;
    const int lane_idx = thread_idx % 32;
    WGRole role = get_wg_role();

    TileScheduler tile_scheduler{params.tile_scheduler_params};

    extern __shared__ char shared_buf[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(shared_buf);

    int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
    int* cu_len_ptr = (int*)params.cu_seqlens_ptr;

    // ── Barrier initialization (WG0 / warp0 / elected thread) ──
    if (wg_idx == 0 && warp_idx_in_wg == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dAqk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dAkk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dq.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dg.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_beta.get_tma_descriptor());

        // TmaAsync barriers: LdSt → Prep (arrival count = 1 for TMA)
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_qkg_ready[i], 1);
        }
        cute::initialize_barrier(smem->bar_dA_ready, 2);  // TMA (arrive_expect_tx) + warp1 explicit arrive

        // Async barriers: Prep → MMA (arrival count = 256 for Prep WGs)
        cute::initialize_barrier(smem->bar_mask_ready, 256);
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_kg_ready[i], 256);
            cute::initialize_barrier(smem->bar_qg_ready[i], 256);
            cute::initialize_barrier(smem->bar_kbg_ready[i], 256);
        }

        // Async barriers: MMA → Prep (arrival count = 128 for MMA WG)
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_mma_ki_ready[i], 128);
        }

        // Prep → LdSt
        cute::initialize_barrier(smem->bar_epilogue_done, 256);

        // Buffer free signals (arrival count = 256 for Prep)
        for (int i = 0; i < NUM_BUF; ++i) {
            cute::initialize_barrier(smem->bar_buf_free[i], 256);
        }

        cutlass::arch::fence_barrier_init();
    }

    __syncthreads();

    // ── Phase tracking ──
    // Double-buffered barriers need per-buffer phase tracking because
    // each buffer's barrier has an independent phase counter.
    int buf_idx = 0;                     // double-buffer index for ki data
    int phase_qkg[NUM_BUF] = {0, 0};     // barrier phase for qkg_pipeline (Q,K,G + dQ,dK,dG)
    int phase_dA = 0;                    // barrier phase for dA_pipeline (single)
    int phase_mask = 0;                  // barrier phase for mask_pipeline (single)
    int phase_kg[NUM_BUF] = {0, 0};      // barrier phase for kg_pipeline
    int phase_qg[NUM_BUF] = {0, 0};      // barrier phase for qg_pipeline
    int phase_kbg[NUM_BUF] = {0, 0};     // barrier phase for kbg_pipeline
    int phase_mma_ki[NUM_BUF] = {0, 0};  // barrier phase for mma_ki_pipeline
    int phase_epi = 0;                   // barrier phase for epilogue_pipeline (single)
    int phase_free[NUM_BUF] = {0, 0};    // barrier phase for buf_free

    // =================================================================
    // WG0: LdSt — TMA loads + writes
    // =================================================================
    if (role == WGRole::LdSt) {
        cutlass::arch::warpgroup_reg_dealloc<REG_LDST>();

        // ── Warp0: TMA load Q, K, G + dQ, dK, dG per ki (double-buffered) ──
        if (warp_idx_in_wg == 0 && elect_one_sync()) {
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);
                int token_offset = cu_len_ptr[batch_idx];

                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    int cur_buf = k_idx % NUM_BUF;

                    // Wait for buffer to be free (from previous tile's Prep epilogue)
                    if (k_idx >= NUM_BUF) {
                        cute::wait_barrier(smem->bar_buf_free[cur_buf], phase_free[cur_buf]);
                        phase_free[cur_buf] ^= 1;
                    }

                    // TMA load Q, K, G + dQ, dK, dG inter-chunk (single barrier)
                    {
                        Tensor sQ = make_tensor(make_smem_ptr(smem->smem_q[cur_buf].data()), SmemLayoutInputBF16{});
                        Tensor sK = make_tensor(make_smem_ptr(smem->smem_k[cur_buf].data()), SmemLayoutInputBF16{});
                        Tensor sG = make_tensor(make_smem_ptr(smem->smem_g[cur_buf].data()), SmemLayoutInputFP32{});
                        Tensor sDQ =
                            make_tensor(make_smem_ptr(smem->smem_dq_in[cur_buf].data()), SmemLayoutInputFP32{});
                        Tensor sDK =
                            make_tensor(make_smem_ptr(smem->smem_dk_in[cur_buf].data()), SmemLayoutInputFP32{});
                        Tensor sDG =
                            make_tensor(make_smem_ptr(smem->smem_dg_in[cur_buf].data()), SmemLayoutInputFP32{});

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

                        cute::set_barrier_transaction_bytes(smem->bar_qkg_ready[cur_buf], tma_bytes);
                        launch_tma_copy(tma_params.tma_q, gQ, sQ, smem->bar_qkg_ready[cur_buf]);
                        launch_tma_copy(tma_params.tma_k, gK, sK, smem->bar_qkg_ready[cur_buf]);
                        launch_tma_copy(tma_params.tma_g, gG, sG, smem->bar_qkg_ready[cur_buf]);
                        launch_tma_copy(tma_params.tma_dq, gDQ, sDQ, smem->bar_qkg_ready[cur_buf]);
                        launch_tma_copy(tma_params.tma_dk, gDK, sDK, smem->bar_qkg_ready[cur_buf]);
                        launch_tma_copy(tma_params.tma_dg, gDG, sDG, smem->bar_qkg_ready[cur_buf]);
                    }
                }  // end per-ki TMA loads
            }  // end persistent tile loop
        }

        // ── Warp1: TMA load dAqk, dAkk + beta (once per tile) ──
        if (warp_idx_in_wg == 1 && elect_one_sync()) {
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);
                int token_offset = cu_len_ptr[batch_idx];

                // TMA load dAqk, dAkk
                {
                    Tensor sDAqk = make_tensor(make_smem_ptr(smem->smem_daqk.data()), SmemLayoutDA{});
                    Tensor sDAkk = make_tensor(make_smem_ptr(smem->smem_dakk.data()), SmemLayoutDA{});
                    int tma_bytes_da = sizeof(make_tensor_like(sDAqk)) + sizeof(make_tensor_like(sDAkk));

                    Tensor mDaqk = domain_offset(
                        make_coord(token_offset, _0{}, _0{}), tma_params.tma_dAqk.get_tma_tensor(tma_params.shape_da));
                    Tensor mDakk = domain_offset(
                        make_coord(token_offset, _0{}, _0{}), tma_params.tma_dAkk.get_tma_tensor(tma_params.shape_da));
                    Tensor gDaqk = local_tile(
                        mDaqk(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<T_TILE>{}), make_coord(tile_idx, _0{}));
                    Tensor gDakk = local_tile(
                        mDakk(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<T_TILE>{}), make_coord(tile_idx, _0{}));

                    cute::set_barrier_transaction_bytes(smem->bar_dA_ready, tma_bytes_da);
                    launch_tma_copy(tma_params.tma_dAqk, gDaqk, sDAqk, smem->bar_dA_ready);
                    launch_tma_copy(tma_params.tma_dAkk, gDakk, sDAkk, smem->bar_dA_ready);
                }

                // Load beta (scalar, via elected thread direct global load)
                {
                    int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
                    int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);
                    float* beta_base = (float*)params.beta_ptr;
                    for (int i = 0; i < T_TILE; ++i) {
                        smem->beta_smem[i] =
                            (i < sub_seq_len) ? beta_base[(token_offset + tile_idx * T_TILE + i) * params.h + head_idx]
                                              : 0.0f;
                    }
                    fence_view_async_shared();
                }

                // Signal Prep: dA + beta ready (pairs with bar_dA_ready arrival count = 2)
                cute::arrive_barrier(smem->bar_dA_ready);
            }  // end persistent tile loop
        }

        // Warp2: idle
        // Warp3: Phase 4 TMA store dB

        // =================================================================
        // WG1: MMA — mma.sync 4-pass sub-chunk loop (Phase 1: placeholder)
        // =================================================================
    } else if (role == WGRole::MMA) {
        cutlass::arch::warpgroup_reg_alloc<REG_MMA>();

        // Phase 1: MMA WG waits for dA to be ready, then does nothing
        // (Phase 3 will implement the actual 4-pass mma.sync loop)
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            // Wait for Prep to signal dA data ready
            cute::wait_barrier(smem->bar_mask_ready, phase_mask);

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int tile_idx = get<2>(blk_coord);
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

            for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                int cur_buf = k_idx % NUM_BUF;

                // Wait for KG ready
                cute::wait_barrier(smem->bar_kg_ready[cur_buf], phase_kg[cur_buf]);

                // Wait for QG ready
                cute::wait_barrier(smem->bar_qg_ready[cur_buf], phase_qg[cur_buf]);

                // Wait for KBG ready
                cute::wait_barrier(smem->bar_kbg_ready[cur_buf], phase_kbg[cur_buf]);

                // Phase 1 placeholder: just write zeros to dQ/dK output buffers
                // TODO(Phase 3): Implement 4-pass mma.sync sub-chunk loop
                for (int elem = idx_in_wg; elem < T_TILE * K_TILE; elem += 128) {
                    smem->smem_dq_out[cur_buf][elem] = 0.0f;
                    smem->smem_dk_out[cur_buf][elem] = 0.0f;
                }
                fence_view_async_shared();

                // Signal Prep: dQ+dK for this ki ready
                cute::arrive_barrier(smem->bar_mma_ki_ready[cur_buf]);

                phase_kg[cur_buf] ^= 1;
                phase_qg[cur_buf] ^= 1;
                phase_kbg[cur_buf] ^= 1;
            }

            phase_mask ^= 1;
        }

        // =================================================================
        // WG2 & WG3: Prep — scalar computation (Phase 1: skeleton)
        // =================================================================
    } else {
        cutlass::arch::warpgroup_reg_alloc<REG_PREP>();

        // Prep thread ID within the 256-thread Prep group
        int prep_tid = thread_idx - 256;  // [0, 255]

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

            // ── Notify MMA: raw dA data ready (mma.sync will do on-the-fly mask) ──
            fence_view_async_shared();
            cute::arrive_barrier(smem->bar_mask_ready);

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

                // ── Construct KG: KG[j,d] = K[j,d] * exp2f(G_norm[d] - G[j,d]) ──
                // Per-subchunk G_norm: 4 sub-tiles, each 16 rows
                {
                    float* kg_ptr = smem->smem_kg[cur_buf].data();
                    float* g_ptr = smem->smem_g[cur_buf].data();
                    bf16* k_ptr = reinterpret_cast<bf16*>(smem->smem_k[cur_buf].data());

                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int j = elem / K_TILE;
                        int d = elem % K_TILE;
                        int js = j / SUB_T_TILE;           // sub-tile index
                        int g_norm_row = js * SUB_T_TILE;  // G_norm = G[js*16, d]

                        float g_val = g_ptr[j * K_TILE + d];
                        float g_norm = g_ptr[g_norm_row * K_TILE + d];
                        float k_val = static_cast<float>(k_ptr[j * K_TILE + d]);

                        kg_ptr[j * K_TILE + d] = k_val * exp2f(g_norm - g_val);
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
                        int is = i / SUB_T_TILE;
                        int g_norm_row = is * SUB_T_TILE;

                        float g_val = g_ptr[i * K_TILE + d];
                        float g_norm = g_ptr[g_norm_row * K_TILE + d];
                        float q_val = static_cast<float>(q_ptr[i * K_TILE + d]);

                        qg_ptr[i * K_TILE + d] = q_val * exp2f(g_val - g_norm);
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
                        int is = i / SUB_T_TILE;
                        int g_norm_row = is * SUB_T_TILE;

                        float g_val = g_ptr[i * K_TILE + d];
                        float g_norm = g_ptr[g_norm_row * K_TILE + d];
                        float k_val = static_cast<float>(k_ptr[i * K_TILE + d]);
                        float beta_val = smem->beta_smem[i];

                        kbg_ptr[i * K_TILE + d] = k_val * beta_val * exp2f(g_val - g_norm);
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

                // ── Epilogue per ki (Phase 1: placeholder — just pass through) ──
                // Q, K, G + dQ, dK, dG inter already available (waited bar_qkg_ready above)

                // TODO(Phase 4): Full epilogue implementation
                // For now: write dQ_inter, dK_inter, dG_inter directly to output (pass-through)
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

                    // Each Prep thread handles a subset of elements
                    for (int elem = prep_tid; elem < T_TILE * K_TILE; elem += 256) {
                        int row = elem / K_TILE;
                        int col = elem % K_TILE;
                        if (row < sub_seq_len) {
                            // Phase 1: output = inter (MMA output is zeros placeholder)
                            float dq_intra = smem->smem_dq_out[cur_buf][row * K_TILE + col];
                            float dk_intra = smem->smem_dk_out[cur_buf][row * K_TILE + col];
                            float dq_inter = smem->smem_dq_in[cur_buf][row * K_TILE + col];
                            float dk_inter = smem->smem_dk_in[cur_buf][row * K_TILE + col];
                            float dg_inter = smem->smem_dg_in[cur_buf][row * K_TILE + col];

                            // dQ_final = dQ_inter + dQ_intra
                            dq_out_base[row * stride_hd + col] = dq_inter + dq_intra;
                            // dK_final = dK_inter + dK_intra (simplified in Phase 1)
                            dk_out_base[row * stride_hd + col] = dk_inter + dk_intra;
                            // dG_final = dG_inter (Phase 1 placeholder)
                            dg_out_base[row * stride_hd + col] = dg_inter;
                        }
                    }
                }

                // Signal buffer free for next ki TMA load
                cute::arrive_barrier(smem->bar_buf_free[cur_buf]);

                phase_qkg[cur_buf] ^= 1;
                phase_mma_ki[cur_buf] ^= 1;
            }  // end per-ki loop

            // ── dB output (Phase 1: write zeros) ──
            // TODO(Phase 4): accumulate dB across ki, then store
            if (prep_tid < T_TILE && prep_tid < sub_seq_len) {
                float* db_out =
                    (float*)params.db_out_ptr + (token_offset + tile_idx * T_TILE + prep_tid) * params.h + head_idx;
                *db_out = smem->db_accum[prep_tid];
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

    // Beta TMA: shape [total_q_len, H], stride [H, 1]
    auto shape_Beta = make_shape(total_q_len, H);
    auto stride_Beta = make_stride(H, _1{});
    auto tma_Beta = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.beta_ptr), make_layout(shape_Beta, stride_Beta)),
        SmemLayoutBeta{});

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
        decltype(tma_DG),
        decltype(tma_Beta)>{
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
        tma_Beta,
    };

    auto kda_kernel = &kda_bwd_intra_sm90_kernel<decltype(tma_params_val)>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    CHECK_CUDA(cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(NUM_THREADS, 1, 1);
    kda_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params_val);
}

}  // namespace sm90_bwd
