#include <math.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

#include "kda_config.h"

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
using SmemLayoutA = Layout<Shape<_16, _16>, Stride<_16, _1>>;                 // dA: [BC,BC]
using SmemLayoutB = Layout<Shape<Int<32>, _16>, Stride<_16, _1>>;             // B:  [BK,BC]
using SmemLayout_QKG = Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>;  // [16,32]

// G2S TiledCopy: 128 threads, 128-bit / 64-bit fp32 loads
using G2SCopy_16x32 = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, float>{}, Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _4>>{}));

using G2SCopy_16x16 = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint64_t>, float>{}, Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _2>>{}));

// Thread layout for bf16→fp32 local_partition
using ThrLayout_16x8 = Layout<Shape<_16, _8>, Stride<_8, _1>>;

// S2R atom for MMA operand loads
using S2RAtom = Copy_Atom<UniversalCopy<float>, float>;

// ============================================================
// Shared memory
// ============================================================
struct SmemStorage {
    float s_q[BC * BK];      // [16, 32] persistent
    float s_k[BC * BK];      // [16, 32] persistent
    float s_g[BC * BK];      // [16, 32] persistent
    float s_beta[BC];        // [16]     persistent
    float s_gn[BK];          // [32]     gate anchor
    float s_dA_qk[BC * BC];  // [16, 16] per-iter
    float s_dA_kk[BC * BC];  // [16, 16] per-iter
    float s_B[BK * BC];      // [32, 16] B operand
    union {
        float s_B2[BK * BC];   // Phase 2: second B operand
        float s_acc[BC * BK];  // Phase 1: db reduction scratch
    };
    float s_db[BC];  // [16] db output
};

// ============================================================
// Helper: create a [R,C] subtile from a 2D gmem tensor at (r,c)
// ============================================================
template <int R, int C, class Engine, class Layout>
__device__ __forceinline__ auto
gmem_subtile(Tensor<Engine, Layout> const& t, int r, int c = 0) {
    return make_tensor(t.data() + t.layout()(r, c), make_shape(Int<R>{}, Int<C>{}), t.stride());
}

// ============================================================
// G2S: bf16 gmem → fp32 smem via local_partition + convert
// ============================================================
template <class SmemTensor, class GmemTensor>
__device__ __forceinline__ void
g2s_bf16_to_f32(SmemTensor& sT, GmemTensor const& gT, int tid) {
    auto tG = local_partition(gT, ThrLayout_16x8{}, tid);
    auto tS = local_partition(sT, ThrLayout_16x8{}, tid);
    CUTE_UNROLL
    for (int i = 0; i < size(tS); ++i) {
        tS(i) = __bfloat162float(tG(i));
    }
}

// G2S: fp32 gmem → fp32 smem via TiledCopy [16,32]
template <class SmemTensor, class GmemTensor>
__device__ __forceinline__ void
g2s_f32_16x32(SmemTensor& sT, GmemTensor const& gT, int tid) {
    G2SCopy_16x32 g2s;
    auto thr = g2s.get_slice(tid);
    copy(g2s, thr.partition_S(gT), thr.partition_D(sT));
}

// G2S: fp32 gmem → fp32 smem via TiledCopy [16,16]
template <class SmemTensor, class GmemTensor>
__device__ __forceinline__ void
g2s_f32_16x16(SmemTensor& sT, GmemTensor const& gT, int tid) {
    G2SCopy_16x16 g2s;
    auto thr = g2s.get_slice(tid);
    copy(g2s, thr.partition_S(gT), thr.partition_D(sT));
}

// ============================================================
// MMA helper: acc += A[16,16] @ B[32,16]^T → C[16,32]
// Follows sgemm_sm80 pattern: s2r copy with retile_D
// ============================================================
__device__ __forceinline__ void
gemm_m16n32k16(const float* s_A, const float* s_B, float acc[4], int tid) {
    TiledMMA_t tiled_mma;
    auto sA = make_tensor(make_smem_ptr(s_A), SmemLayoutA{});  // (16,16)
    auto sB = make_tensor(make_smem_ptr(s_B), SmemLayoutB{});  // (32,16)

    auto thr_mma = tiled_mma.get_slice(tid);
    auto tCrA = thr_mma.partition_fragment_A(sA);  // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(sB);  // (MMA, MMA_N, MMA_K)

    // Accumulator: create using a [16,32] shape for C
    auto sC_dummy = make_tensor(make_smem_ptr(s_A), Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{});
    auto tCrC = thr_mma.partition_fragment_C(sC_dummy);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i)
        tCrC(i) = acc[i];

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
    for (int i = 0; i < size(tCrC); ++i)
        acc[i] = tCrC(i);
}

// ============================================================
// Per-thread accumulator → (row, col) mapping
// SM80_16x8x8 CLayout: row = (lane/8)*4+v, col = (lane%8)+warp_id*8
// ============================================================
__device__ __forceinline__ void
get_acc_row_col(int tid, int v, int& row, int& col) {
    int lane = tid % 32;
    int warp_id = tid / 32;
    row = (lane / 8) * 4 + v;
    col = (lane % 8) + warp_id * 8;
}

// ============================================================
// Main kernel
// ============================================================
__global__ void
__launch_bounds__(NUM_THREADS) kda_bwd_intra_kernel_sm90(const KDA_bwd_intra_params params) {
    extern __shared__ char smem_buf[];
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(smem_buf);

    // ── Extract typed pointers from params ──
    const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q_ptr);
    const auto* k_ptr = reinterpret_cast<const __nv_bfloat16*>(params.k_ptr);
    const auto* g_ptr = reinterpret_cast<const float*>(params.g_ptr);
    const auto* beta_ptr = reinterpret_cast<const __nv_bfloat16*>(params.beta_ptr);
    const auto* dAqk_ptr = reinterpret_cast<const float*>(params.dAqk_ptr);
    const auto* dAkk_ptr = reinterpret_cast<const float*>(params.dAkk_ptr);
    const auto* dq_ptr = reinterpret_cast<const float*>(params.dq_ptr);
    const auto* dk_ptr = reinterpret_cast<const float*>(params.dk_ptr);
    const auto* dg_ptr = reinterpret_cast<const float*>(params.dg_ptr);
    auto* dq_out_ptr = reinterpret_cast<__nv_bfloat16*>(params.dq_out_ptr);
    auto* dk_out_ptr = reinterpret_cast<__nv_bfloat16*>(params.dk_out_ptr);
    auto* db_out_ptr = reinterpret_cast<float*>(params.db_out_ptr);
    auto* dg_out_ptr = reinterpret_cast<float*>(params.dg_out_ptr);
    const auto* cu_seqlens = reinterpret_cast<const int*>(params.cu_seqlens_ptr);
    const auto* chunk_idx = reinterpret_cast<const int*>(params.chunk_indices_ptr);

    const int H = params.h;
    const int K = params.d;
    const int total_q_len = params.total_q_len;

    const int tid = threadIdx.x;

    // Decode block coordinates via tile scheduler
    NaiveTileScheduler scheduler(params.tile_scheduler_params);
    auto coord = scheduler.get_block_coord(cu_seqlens, chunk_idx);
    // coord: (i_n, i_h, i_t, i_k, i_i, bos, T_seq)
    // head idx
    const int i_h = get<1>(coord);
    // current tile index within the sequence
    const int i_t = get<2>(coord);
    // k-block idx for the current tile
    const int i_k = get<3>(coord);
    // sub-tile idx for the current tile (0 to NC-1)
    const int i_i = get<4>(coord);
    // batch offset for the current sequence
    const int bos = get<5>(coord);
    // sequence length for the current batch (eos - bos)
    const int T_seq = get<6>(coord);

    // BT: chunk size; BC: tile size. i_i is the subchunk index within the chunk, used for indexing dA and gating
    // outputs.
    const int i_ti = i_t * BT + i_i * BC;
    if (i_ti >= T_seq)
        return;

    const int HK = H * K;
    const int H_BT = H * BT;

    // ── CuTe global memory tensor views ──
    // Helper: offset into [total_q_len, H, K] layout → row bos, head i_h, k-block i_k
    const auto off_hk = bos * HK + i_h * K + i_k * BK;

    auto mQ = make_tensor(make_gmem_ptr(q_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mK = make_tensor(make_gmem_ptr(k_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mG = make_tensor(make_gmem_ptr(g_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mBeta = make_tensor(make_gmem_ptr(beta_ptr + bos * H + i_h), make_shape(T_seq), make_stride(H));
    auto mDAqk = make_tensor(
        make_gmem_ptr(dAqk_ptr + bos * H_BT + i_h * BT), make_shape(T_seq, Int<BT>{}), make_stride(H_BT, Int<1>{}));
    auto mDAkk = make_tensor(
        make_gmem_ptr(dAkk_ptr + bos * H_BT + i_h * BT), make_shape(T_seq, Int<BT>{}), make_stride(H_BT, Int<1>{}));
    auto mDq = make_tensor(make_gmem_ptr(dq_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mDk = make_tensor(make_gmem_ptr(dk_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mDg = make_tensor(make_gmem_ptr(dg_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mDqOut =
        make_tensor(make_gmem_ptr(dq_out_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mDkOut =
        make_tensor(make_gmem_ptr(dk_out_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mDgOut =
        make_tensor(make_gmem_ptr(dg_out_ptr + off_hk), make_shape(T_seq, Int<BK>{}), make_stride(HK, Int<1>{}));
    auto mDBout = make_tensor(
        make_gmem_ptr(db_out_ptr + i_k * total_q_len * H + bos * H + i_h), make_shape(T_seq), make_stride(H));

    // ── SMEM tensor views ──
    auto sQ = make_tensor(make_smem_ptr(smem.s_q), SmemLayout_QKG{});
    auto sK = make_tensor(make_smem_ptr(smem.s_k), SmemLayout_QKG{});
    auto sG = make_tensor(make_smem_ptr(smem.s_g), SmemLayout_QKG{});
    auto sB = make_tensor(make_smem_ptr(smem.s_B), SmemLayoutB{});
    auto sDAqk = make_tensor(make_smem_ptr(smem.s_dA_qk), SmemLayoutA{});
    auto sDAkk = make_tensor(make_smem_ptr(smem.s_dA_kk), SmemLayoutA{});

    // ── Load persistent tiles using CuTe G2S ──
    {
        auto gQ_tile = gmem_subtile<BC, BK>(mQ, i_ti);
        auto gK_tile = gmem_subtile<BC, BK>(mK, i_ti);
        auto gG_tile = gmem_subtile<BC, BK>(mG, i_ti);
        g2s_bf16_to_f32(sQ, gQ_tile, tid);
        g2s_bf16_to_f32(sK, gK_tile, tid);
        g2s_f32_16x32(sG, gG_tile, tid);
    }
    if (tid < BC) {
        smem.s_beta[tid] = __bfloat162float(mBeta(i_ti + tid));
    }
    __syncthreads();

    // Per-thread accumulators
    float dq2_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float dk2_acc[4] = {0.f, 0.f, 0.f, 0.f};

    // ════════════════════════════════════════════
    // PHASE 1: Off-diagonal (j < i_i)
    // ════════════════════════════════════════════
    if (i_i > 0) {
        if (tid < BK) {
            smem.s_gn[tid] = mG(i_ti, tid);
        }
        __syncthreads();

        for (int i_j = 0; i_j < i_i; ++i_j) {
            int j_off = i_t * BT + i_j * BC;

            // Load dA tiles [BC, BC]
            {
                auto gDAqk_tile = gmem_subtile<BC, BC>(mDAqk, i_ti, i_j * BC);
                auto gDAkk_tile = gmem_subtile<BC, BC>(mDAkk, i_ti, i_j * BC);
                g2s_f32_16x16(sDAqk, gDAqk_tile, tid);
                g2s_f32_16x16(sDAkk, gDAkk_tile, tid);
            }

            // Build B[n,k] = k_j[k,n] * exp2(gn[n] - g_j[k,n])
            for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
                int n = idx / BC;
                int k_dim = idx % BC;
                float kv = __bfloat162float(mK(j_off + k_dim, n));
                float gv = mG(j_off + k_dim, n);
                sB(n, k_dim) = kv * exp2f(smem.s_gn[n] - gv);
            }
            __syncthreads();

            gemm_m16n32k16(smem.s_dA_qk, smem.s_B, dq2_acc, tid);
            gemm_m16n32k16(smem.s_dA_kk, smem.s_B, dk2_acc, tid);
            __syncthreads();
        }

        // Post-multiply: *= exp2(g - gn)
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            float scale = exp2f(sG(row, col) - smem.s_gn[col]);
            dq2_acc[v] *= scale;
            dk2_acc[v] *= scale;
        }
    }

    // ════════════════════════════════════════════
    // PHASE 1: Diagonal (safe_gate)
    // ════════════════════════════════════════════
    {
        int mid = min(BC / 2, T_seq - i_ti - 1);
        if (tid < BK) {
            smem.s_gn[tid] = mG(i_ti + mid, tid);
        }
        __syncthreads();

        {
            auto gDAqk_tile = gmem_subtile<BC, BC>(mDAqk, i_ti, i_i * BC);
            auto gDAkk_tile = gmem_subtile<BC, BC>(mDAkk, i_ti, i_i * BC);
            g2s_f32_16x16(sDAqk, gDAqk_tile, tid);
            g2s_f32_16x16(sDAkk, gDAkk_tile, tid);
        }
        __syncthreads();

        // Apply lower-triangular mask in SMEM
        for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
            int r = idx / BC, c = idx % BC;
            bool valid = (r >= c) && (i_ti + r < T_seq) && (i_ti + c < T_seq);
            if (!valid) {
                sDAqk(r, c) = 0.f;
                sDAkk(r, c) = 0.f;
            }
        }
        __syncthreads();

        // B = k * exp2(-(g - gn))
        for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
            int n = idx / BC;
            int k_dim = idx % BC;
            bool valid = (i_ti + k_dim) < T_seq;
            float g_diff = valid ? (sG(k_dim, n) - smem.s_gn[n]) : 0.f;
            sB(n, k_dim) = valid ? (sK(k_dim, n) * exp2f(-g_diff)) : 0.f;
        }
        __syncthreads();

        float tmp_dq[4] = {0.f, 0.f, 0.f, 0.f};
        float tmp_dk[4] = {0.f, 0.f, 0.f, 0.f};
        gemm_m16n32k16(smem.s_dA_qk, smem.s_B, tmp_dq, tid);
        gemm_m16n32k16(smem.s_dA_kk, smem.s_B, tmp_dk, tid);

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

    // ════════════════════════════════════════════
    // PHASE 1: Epilogue
    // ════════════════════════════════════════════
    __syncthreads();
    float dg2_acc[4];
    {
        // db = sum(dk2 * k, dim=K) via smem scratch + warp shuffle
        auto sAcc = make_tensor(make_smem_ptr(smem.s_acc), SmemLayout_QKG{});
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            sAcc(row, col) = dk2_acc[v] * sK(row, col);
        }
        __syncthreads();

        // Warp-level row reduction: BK=32 == warp_size
        {
            int warp_id = tid / 32;
            int lane = tid % 32;
            for (int pass = 0; pass < 4; ++pass) {
                int row = pass * 4 + warp_id;
                if (row < BC) {
                    float val = sAcc(row, lane);
                    for (int offset = 16; offset > 0; offset >>= 1)
                        val += __shfl_xor_sync(0xffffffff, val, offset);
                    if (lane == 0)
                        smem.s_db[row] = val;
                }
            }
        }
        __syncthreads();

        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            dg2_acc[v] = sQ(row, col) * dq2_acc[v];
            dk2_acc[v] *= smem.s_beta[row];
            dq2_acc[v] += mDq(i_ti + row, col);
        }

        // Store dq_out (bf16), db_out (fp32)
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            if ((i_ti + row) < T_seq) {
                mDqOut(i_ti + row, col) = __float2bfloat16(dq2_acc[v]);
            }
        }
        if (tid < BC && (i_ti + tid) < T_seq) {
            mDBout(i_ti + tid) = smem.s_db[tid];
        }
    }

    // ════════════════════════════════════════════
    // PHASE 2: Off-diagonal (j > i_i)
    // ════════════════════════════════════════════
    __syncthreads();
    float dkt_acc[4] = {0.f, 0.f, 0.f, 0.f};

    int NC_eff = min(NC, (T_seq - i_t * BT + BC - 1) / BC);
    if (i_i < NC_eff - 1) {
        int last_row = min((int)(i_ti + BC), T_seq) - 1;
        if (tid < BK) {
            smem.s_gn[tid] = mG(last_row, tid);
        }
        __syncthreads();

        for (int i_j = i_i + 1; i_j < NC_eff; ++i_j) {
            int j_off = i_t * BT + i_j * BC;

            // Load transposed dA: smem[r,c] = dA[j_off+c, i_i*BC+r]
            for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
                int r = idx / BC, c = idx % BC;
                sDAqk(r, c) = mDAqk(j_off + c, i_i * BC + r);
                sDAkk(r, c) = mDAkk(j_off + c, i_i * BC + r);
            }

            // B1 = q_j*exp2(g_j-gn), B2 = k_j*beta_j*exp2(g_j-gn)
            {
                auto sB2 = make_tensor(make_smem_ptr(smem.s_B2), SmemLayoutB{});
                for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
                    int n = idx / BC;
                    int k_dim = idx % BC;
                    int row_idx = j_off + k_dim;
                    bool valid = row_idx < T_seq;
                    float qv = valid ? __bfloat162float(mQ(row_idx, n)) : 0.f;
                    float kv = valid ? __bfloat162float(mK(row_idx, n)) : 0.f;
                    float gv = valid ? mG(row_idx, n) : 0.f;
                    float bv = valid ? __bfloat162float(mBeta(row_idx)) : 0.f;
                    float scale = valid ? exp2f(gv - smem.s_gn[n]) : 0.f;
                    sB(n, k_dim) = qv * scale;
                    sB2(n, k_dim) = kv * bv * scale;
                }
            }
            __syncthreads();

            gemm_m16n32k16(smem.s_dA_qk, smem.s_B, dkt_acc, tid);
            gemm_m16n32k16(smem.s_dA_kk, smem.s_B2, dkt_acc, tid);
            __syncthreads();
        }

        // Post-multiply: *= exp2(gn - g)
        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            dkt_acc[v] *= exp2f(smem.s_gn[col] - sG(row, col));
        }
    }

    // ════════════════════════════════════════════
    // PHASE 2: Diagonal (safe_gate)
    // ════════════════════════════════════════════
    __syncthreads();
    {
        int mid = min(BC / 2, T_seq - i_ti - 1);
        if (tid < BK) {
            smem.s_gn[tid] = mG(i_ti + mid, tid);
        }
        __syncthreads();

        // Load transposed dA with upper-tri mask
        for (int idx = tid; idx < BC * BC; idx += NUM_THREADS) {
            int r = idx / BC, c = idx % BC;
            float daqk = mDAqk(i_ti + c, i_i * BC + r);
            float dakk = mDAkk(i_ti + c, i_i * BC + r);
            bool valid = (r <= c) && (i_ti + r < T_seq) && (i_ti + c < T_seq);
            sDAqk(r, c) = valid ? daqk : 0.f;
            sDAkk(r, c) = valid ? dakk : 0.f;
        }

        // B1 = q*exp2(g-gn), B2 = k*beta*exp2(g-gn) from persistent smem
        {
            auto sB2 = make_tensor(make_smem_ptr(smem.s_B2), SmemLayoutB{});
            for (int idx = tid; idx < BK * BC; idx += NUM_THREADS) {
                int n = idx / BC;
                int k_dim = idx % BC;
                bool valid = (i_ti + k_dim) < T_seq;
                float g_diff = valid ? (sG(k_dim, n) - smem.s_gn[n]) : 0.f;
                float scale = valid ? exp2f(g_diff) : 0.f;
                sB(n, k_dim) = sQ(k_dim, n) * scale;
                sB2(n, k_dim) = sK(k_dim, n) * smem.s_beta[k_dim] * scale;
            }
        }
        __syncthreads();

        float tmp_q[4] = {0.f, 0.f, 0.f, 0.f};
        float tmp_k[4] = {0.f, 0.f, 0.f, 0.f};
        gemm_m16n32k16(smem.s_dA_qk, smem.s_B, tmp_q, tid);
        gemm_m16n32k16(smem.s_dA_kk, smem.s_B2, tmp_k, tid);

        for (int v = 0; v < 4; ++v) {
            int row, col;
            get_acc_row_col(tid, v, row, col);
            bool valid = (i_ti + row) < T_seq;
            float g_diff = valid ? (sG(row, col) - smem.s_gn[col]) : 0.f;
            float scale = valid ? exp2f(-g_diff) : 0.f;
            dkt_acc[v] += (tmp_q[v] + tmp_k[v]) * scale;
        }
    }

    // ════════════════════════════════════════════
    // PHASE 2: Epilogue
    // ════════════════════════════════════════════
    for (int v = 0; v < 4; ++v) {
        int row, col;
        get_acc_row_col(tid, v, row, col);
        if ((i_ti + row) < T_seq) {
            float kv = sK(row, col);
            float dg_up = mDg(i_ti + row, col);
            float dk_up = mDk(i_ti + row, col);
            mDgOut(i_ti + row, col) = dg2_acc[v] + (dk2_acc[v] - dkt_acc[v]) * kv + dg_up;
            mDkOut(i_ti + row, col) = __float2bfloat16(dk2_acc[v] + dk_up + dkt_acc[v]);
        }
    }
}

// ============================================================
// Host launch
// ============================================================
namespace sm90 {

void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream) {
    dim3 grid = NaiveTileScheduler::get_grid_shape(params.tile_scheduler_params);
    dim3 block(NUM_THREADS);
    int smem_size = sizeof(SmemStorage);

    kda_bwd_intra_kernel_sm90<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace sm90
