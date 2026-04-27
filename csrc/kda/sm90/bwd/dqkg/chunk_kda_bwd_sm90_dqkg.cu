/// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Phase 2 SM90 CUDA implementation of chunk_kda_bwd_wy_dqkg_fused.
// Uses nvcuda::wmma m16n16k16 bf16→fp32 Tensor Core GEMMs to replace all
// scalar BT×V and BT×K loops (7 GEMMs total).  Phase 5 (BT×BT) stays scalar.
//
// Reference: third_party/flash-linear-attention/fla/ops/kda/chunk_bwd.py
//            chunk_kda_bwd_kernel_wy_dqkg_fused (TRANSPOSE_STATE=False, IS_VARLEN=False)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
#include <math.h>

using namespace nvcuda;

// ---- Compile-time constants ----
static constexpr int NTHREADS  = 128;    // 4 warps
static constexpr int NWARPS    = 4;
static constexpr int BT        = 64;
static constexpr int K_DIM     = 128;
static constexpr int V_DIM     = 128;
static constexpr int WM = 16, WN = 16, WK = 16;

// ============================================================
// Shared memory layout (bytes)
// ============================================================
// OFF_H         =      0 : sm_h[K×V]    bf16  = 32768B  (persistent)
// OFF_DH        =  32768 : sm_dh[K×V]   bf16  = 32768B  (persistent)
// OFF_A         =  65536 : sm_A[BT×BT]  bf16  =  8192B  (persistent through Phase 5)
// OFF_BETA      =  73728 : sm_beta[BT]  f32   =   256B
// OFF_GN        =  73984 : sm_gn[K]     f32   =   512B
// OFF_DGK       =  74496 : sm_dgk[K]    f32   =   512B  (init 0)
// OFF_DB        =  75008 : sm_db[BT]    f32   =   256B  (init 0)
// OFF_DO        =  75264 : sm_do[BT×V]  bf16  = 16384B  → reused as sm_dA[BT×BT] f32 in Phase 5
// OFF_VNEW      =  91648 : sm_vnew[BT×V]bf16  = 16384B  → reused as sm_dA2[BT×BT] f32 in Phase 5
// OFF_DV        = 108032 : sm_dv[BT×V]  bf16  = 16384B  (used by GEMM3, dA GEMM, Phase3)
// OFF_V         = 124416 : sm_v[BT×V]   bf16  = 16384B  (used by dA GEMM, Phase3 db)
// OFF_DK_INNER  = 140800 : sm_dk_inner[BT×K] f32 = 32768B (GEMM1 temp, GEMM3 temp, GEMM2 output)
// OFF_KG        = 173568 : sm_kg[BT×K]  bf16  = 16384B  (scalar Phase1C → dA GEMM, Phase4)
// OFF_DW_NEG    = 189952 : sm_dw_neg[BT×K] bf16 = 16384B (GEMM3 output → dA GEMM, Phase4)
// OFF_WARP_TEMP = 206336 : warp_temp[4×16×16] f32 = 4096B (per-warp inline processing)
// TOTAL         = 210432B ≈ 205 KB  (fits SM90 228 KB limit)

static constexpr int OFF_H          = 0;
static constexpr int OFF_DH         = 32768;
static constexpr int OFF_A          = 65536;
static constexpr int OFF_BETA       = 73728;
static constexpr int OFF_GN         = 73984;
static constexpr int OFF_DGK        = 74496;
static constexpr int OFF_DB         = 75008;
static constexpr int OFF_DO         = 75264;
static constexpr int OFF_VNEW       = 91648;
static constexpr int OFF_DV         = 108032;
static constexpr int OFF_V          = 124416;
static constexpr int OFF_DK_INNER   = 140800;
static constexpr int OFF_KG         = 173568;
static constexpr int OFF_DW_NEG     = 189952;
static constexpr int OFF_WARP_TEMP  = 206336;
static constexpr int SMEM_BYTES     = 210432;

// ---- Kernel ----
// Grid: (NT, B*H)  — one CTA per (chunk, batch-head) pair.
// Block: (NTHREADS=128,) — 4 warps, warp_id = threadIdx.x / 32.
//
// Tensor layouts (non-varlen, TRANSPOSE_STATE=False):
//   q/k/g/dq/dk/dg:     [B*T, H, K]    row-major
//   v/v_new/do_/dv/dv2: [B*T, H, V]    row-major
//   beta/db:             [B*T, H]       row-major
//   A/dA:                [B*T, H, BT]   row-major
//   h/dh:                [NT*B, H, K, V] row-major (K outer, V inner)
__global__ void __launch_bounds__(NTHREADS)
chunk_kda_bwd_dqkg_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const __nv_bfloat16* __restrict__ v_new_ptr,
    const float*          __restrict__ g_ptr,
    const float*          __restrict__ beta_ptr,
    const __nv_bfloat16* __restrict__ A_ptr,
    const __nv_bfloat16* __restrict__ h_ptr,
    const __nv_bfloat16* __restrict__ do_ptr,
    const __nv_bfloat16* __restrict__ dh_ptr,
    const __nv_bfloat16* __restrict__ dv_ptr,
    float*                __restrict__ dq_ptr,
    float*                __restrict__ dk_ptr,
    __nv_bfloat16*        __restrict__ dv2_ptr,
    float*                __restrict__ dg_ptr,
    float*                __restrict__ db_ptr,
    float*                __restrict__ dA_ptr,
    int B, int T, int H,
    float scale)
{
    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int tid  = threadIdx.x;
    const int i_b  = i_bh / H;
    const int i_h  = i_bh % H;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    const int NT          = (T + BT - 1) / BT;
    const int64_t i_tg    = (int64_t)i_b * NT + i_t;
    const int chunk_start = i_t * BT;
    const int chunk_len   = (chunk_start + BT <= T) ? BT : (T - chunk_start);
    const int last_t      = chunk_len - 1;

    // Base offsets into [B*T, H, *] tensors
    const int64_t bos        = (int64_t)i_b * T;
    const int64_t tok0_h     = (bos + chunk_start) * H + i_h;
    const int64_t qk_base    = tok0_h * K_DIM;
    const int64_t v_base     = tok0_h * V_DIM;
    const int64_t beta_base  = tok0_h;
    const int64_t A_base     = tok0_h * BT;
    const int64_t h_base     = (i_tg * H + i_h) * (int64_t)(K_DIM * V_DIM);

    // ---- Shared memory pointers ----
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sm_h        = (__nv_bfloat16*)(smem_raw + OFF_H);
    __nv_bfloat16* sm_dh       = (__nv_bfloat16*)(smem_raw + OFF_DH);
    __nv_bfloat16* sm_A        = (__nv_bfloat16*)(smem_raw + OFF_A);
    float*         sm_beta     = (float*)(smem_raw + OFF_BETA);
    float*         sm_gn       = (float*)(smem_raw + OFF_GN);
    float*         sm_dgk      = (float*)(smem_raw + OFF_DGK);
    float*         sm_db       = (float*)(smem_raw + OFF_DB);
    __nv_bfloat16* sm_do       = (__nv_bfloat16*)(smem_raw + OFF_DO);
    __nv_bfloat16* sm_vnew     = (__nv_bfloat16*)(smem_raw + OFF_VNEW);
    __nv_bfloat16* sm_dv       = (__nv_bfloat16*)(smem_raw + OFF_DV);
    __nv_bfloat16* sm_v        = (__nv_bfloat16*)(smem_raw + OFF_V);
    float*         sm_dk_inner = (float*)(smem_raw + OFF_DK_INNER);
    __nv_bfloat16* sm_kg       = (__nv_bfloat16*)(smem_raw + OFF_KG);
    __nv_bfloat16* sm_dw_neg   = (__nv_bfloat16*)(smem_raw + OFF_DW_NEG);
    float*         sm_warp_temp= (float*)(smem_raw + OFF_WARP_TEMP);  // [4 × WM×WN]

    // Phase 5 reuse pointers (sm_do → sm_dA, sm_vnew → sm_dA2)
    float* sm_dA  = (float*)(smem_raw + OFF_DO);
    float* sm_dA2 = (float*)(smem_raw + OFF_VNEW);

    // Per-warp WMMA temp (WM×WN = 256 floats = 1 KB per warp)
    float* wt = sm_warp_temp + warp_id * (WM * WN);

    // ========================================================
    // Phase 0: Cooperative load of all inputs; init accumulators.
    // ========================================================

    // h, dh: [K_DIM × V_DIM]
    for (int idx = tid; idx < K_DIM * V_DIM; idx += NTHREADS) {
        sm_h[idx]  = h_ptr[h_base + idx];
        sm_dh[idx] = dh_ptr[h_base + idx];
    }

    // A: [BT × BT], rows >= chunk_len zeroed
    for (int idx = tid; idx < BT * BT; idx += NTHREADS) {
        int t_local = idx / BT;
        int j       = idx % BT;
        sm_A[idx] = (t_local < chunk_len)
            ? A_ptr[A_base + (int64_t)t_local * H * BT + j]
            : __float2bfloat16(0.f);
    }

    // beta, gn
    for (int t = tid; t < BT; t += NTHREADS)
        sm_beta[t] = (t < chunk_len) ? beta_ptr[beta_base + (int64_t)t * H] : 0.f;
    for (int kk = tid; kk < K_DIM; kk += NTHREADS)
        sm_gn[kk] = g_ptr[((bos + chunk_start + last_t) * H + i_h) * K_DIM + kk];

    // do, vnew, dv, v: [BT × V_DIM], rows >= chunk_len zeroed
    for (int idx = tid; idx < BT * V_DIM; idx += NTHREADS) {
        int t = idx / V_DIM;
        int v = idx % V_DIM;
        if (t < chunk_len) {
            int64_t vi = v_base + (int64_t)t * H * V_DIM + v;
            sm_do  [idx] = do_ptr  [vi];
            sm_vnew[idx] = v_new_ptr[vi];
            sm_dv  [idx] = dv_ptr  [vi];
            sm_v   [idx] = v_ptr   [vi];
        } else {
            sm_do  [idx] = __float2bfloat16(0.f);
            sm_vnew[idx] = __float2bfloat16(0.f);
            sm_dv  [idx] = __float2bfloat16(0.f);
            sm_v   [idx] = __float2bfloat16(0.f);
        }
    }

    // init accumulators
    for (int kk = tid; kk < K_DIM; kk += NTHREADS) sm_dgk[kk] = 0.f;
    for (int t  = tid; t  < BT;    t  += NTHREADS) sm_db[t]   = 0.f;

    __syncthreads();

    // ========================================================
    // Phase 1C-a: Scalar: compute sm_kg[t,k] and sm_dgk partial (h*dh part).
    //   thread kk = tid (K_DIM = 128 = NTHREADS, each thread owns 1 k-feature)
    //   sm_kg[t*K+kk] = k[t,kk] * exp2(g[t,kk])
    //   sm_dgk[kk]    = exp2(gn[kk]) * sum_v(h[kk,v]*dh[kk,v])
    // ========================================================
    {
        const int kk = tid;
        float dgk_partial = 0.f;
        for (int v = 0; v < V_DIM; v++)
            dgk_partial += __bfloat162float(sm_h[kk * V_DIM + v])
                         * __bfloat162float(sm_dh[kk * V_DIM + v]);
        sm_dgk[kk] = dgk_partial * exp2f(sm_gn[kk]);

        for (int t = 0; t < BT; t++) {
            float g_tk = (t < chunk_len)
                ? g_ptr[((bos + chunk_start + t) * H + i_h) * K_DIM + kk] : 0.f;
            float k_tk = (t < chunk_len)
                ? __bfloat162float(k_ptr[qk_base + (int64_t)t * H * K_DIM + kk]) : 0.f;
            sm_kg[t * K_DIM + kk] = __float2bfloat16(k_tk * exp2f(g_tk));
        }
    }
    __syncthreads();

    // ========================================================
    // GEMM1: dq_raw = sm_do @ sm_h^T   [BT×V] × [K×V]^T → [BT×K]
    //   Warp m_tile = warp_id (row slice), n_tile iterates 0..K/WN-1 (columns of K).
    //   Uses sm_dk_inner as temp fp32 output (32KB).
    //   After __syncthreads: thread-parallel scale + write dq to gmem.
    // ========================================================
    {
        const int m_base = warp_id * WM;
        for (int n_tile = 0; n_tile < K_DIM / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);
            for (int k_tile = 0; k_tile < V_DIM / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                // A: sm_do[m_base, k_base]  row_major  lda = V_DIM
                wmma::load_matrix_sync(a_frag, sm_do + m_base * V_DIM + k_base, V_DIM);
                // B (col_major for h^T): sm_h[n_base*V_DIM + k_base]  ldb = V_DIM
                // col_major: B[r,c] = ptr[c*ldb+r] = sm_h[(n_base+c)*V_DIM+(k_base+r)] = h[n_base+c, k_base+r] = h^T[k_base+r, n_base+c] ✓
                wmma::load_matrix_sync(b_frag, sm_h + n_base * V_DIM + k_base, V_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(sm_dk_inner + m_base * K_DIM + n_base, c_frag,
                                    K_DIM, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Scale dq_raw and write to global dq
    for (int e = tid; e < BT * K_DIM; e += NTHREADS) {
        int t  = e / K_DIM;
        int kk = e % K_DIM;
        if (t < chunk_len) {
            float g_val = g_ptr[((bos + chunk_start + t) * H + i_h) * K_DIM + kk];
            dq_ptr[qk_base + (int64_t)t * H * K_DIM + kk] =
                sm_dk_inner[e] * exp2f(g_val) * scale;
        }
    }
    __syncthreads();

    // ========================================================
    // GEMM3: dw_neg = -(sm_dv @ sm_h^T)  [BT×V] × [K×V]^T → [BT×K]
    //   Reuses sm_dk_inner as temp. After __syncthreads: convert to bf16 and negate → sm_dw_neg.
    // ========================================================
    {
        const int m_base = warp_id * WM;
        for (int n_tile = 0; n_tile < K_DIM / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);
            for (int k_tile = 0; k_tile < V_DIM / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_dv + m_base * V_DIM + k_base, V_DIM);
                wmma::load_matrix_sync(b_frag, sm_h  + n_base * V_DIM + k_base, V_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(sm_dk_inner + m_base * K_DIM + n_base, c_frag,
                                    K_DIM, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Negate and convert to bf16 → sm_dw_neg
    for (int e = tid; e < BT * K_DIM; e += NTHREADS) {
        int t = e / K_DIM;
        sm_dw_neg[e] = __float2bfloat16((t < chunk_len) ? -sm_dk_inner[e] : 0.f);
    }
    __syncthreads();

    // ========================================================
    // GEMM2: dk_inner = sm_vnew @ sm_dh^T  [BT×V] × [K×V]^T → [BT×K]
    //   Stores final scaled dk_inner into sm_dk_inner (fp32, persistent through Phase4).
    // ========================================================
    {
        const int m_base = warp_id * WM;
        for (int n_tile = 0; n_tile < K_DIM / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);
            for (int k_tile = 0; k_tile < V_DIM / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_vnew + m_base * V_DIM + k_base, V_DIM);
                wmma::load_matrix_sync(b_frag, sm_dh   + n_base * V_DIM + k_base, V_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(sm_dk_inner + m_base * K_DIM + n_base, c_frag,
                                    K_DIM, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Scale dk_inner = dk_raw * exp2(gn[k] - g[t,k])
    for (int e = tid; e < BT * K_DIM; e += NTHREADS) {
        int t  = e / K_DIM;
        int kk = e % K_DIM;
        if (t < chunk_len) {
            float g_val = g_ptr[((bos + chunk_start + t) * H + i_h) * K_DIM + kk];
            sm_dk_inner[e] *= exp2f(sm_gn[kk] - g_val);
        } else {
            sm_dk_inner[e] = 0.f;
        }
    }
    __syncthreads();

    // ========================================================
    // Phase 1C-b: Finalize sm_dgk += sum_t(k[t,k] * dk_inner[t,k])
    // ========================================================
    {
        const int kk = tid;
        float sum_kdk = 0.f;
        for (int t = 0; t < chunk_len; t++) {
            float k_tk = __bfloat162float(k_ptr[qk_base + (int64_t)t * H * K_DIM + kk]);
            sum_kdk += k_tk * sm_dk_inner[t * K_DIM + kk];
        }
        sm_dgk[kk] += sum_kdk;
    }
    __syncthreads();

    // ========================================================
    // dA GEMMs: sm_dA[i,j] = (sm_dv @ sm_v^T)[i,j] + (sm_dw_neg @ sm_kg^T)[i,j]
    //   Both are [BT×BT], m_tile=warp_id, n_tile=0..BT/WN-1.
    //   Combined in one accumulator pass per (m,n) tile.
    //   Output written to sm_dA = (float*)(smem_raw + OFF_DO).
    // ========================================================
    {
        const int m_base = warp_id * WM;
        for (int n_tile = 0; n_tile < BT / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);

            // Pass 1: sm_dv[BT×V] @ sm_v[BT×V]^T
            for (int k_tile = 0; k_tile < V_DIM / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_dv + m_base * V_DIM + k_base, V_DIM);
                // col_major: b_frag[r,c] = sm_v[(n_base+c)*V_DIM + k_base+r] = v^T[k_base+r, n_base+c] ✓
                wmma::load_matrix_sync(b_frag, sm_v  + n_base * V_DIM + k_base, V_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // Pass 2: sm_dw_neg[BT×K] @ sm_kg[BT×K]^T
            for (int k_tile = 0; k_tile < K_DIM / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_dw_neg + m_base * K_DIM + k_base, K_DIM);
                wmma::load_matrix_sync(b_frag, sm_kg     + n_base * K_DIM + k_base, K_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            wmma::store_matrix_sync(sm_dA + m_base * BT + n_base, c_frag,
                                    BT, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // ========================================================
    // Phase 3: dvb = sm_A^T @ sm_dv   [BT×BT]^T × [BT×V] → [BT×V]
    //   Inline: write dv2[j,v] = dvb[j,v]*beta[j]; accumulate db[j] += dvb[j,v]*v[j,v].
    //   Uses sm_warp_temp (1KB per warp) for the 16×16 WMMA output tiles.
    //   A^T loaded col_major: a_frag[r,c] = sm_A[(k_base+c)*BT + m_base+r] = A^T[m_base+r, k_base+c] ✓
    // ========================================================
    {
        const int m_base = warp_id * WM;
        float db_local[WM] = {};   // accumulate db over all n_tiles

        for (int n_tile = 0; n_tile < V_DIM / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);

            for (int k_tile = 0; k_tile < BT / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                // A^T col_major: ptr = sm_A + k_base*BT + m_base, lda = BT
                wmma::load_matrix_sync(a_frag, sm_A  + k_base * BT + m_base, BT);
                wmma::load_matrix_sync(b_frag, sm_dv + k_base * V_DIM + n_base, V_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(wt, c_frag, WN, wmma::mem_row_major);
            // wt[r*WN + c] = dvb[m_base+r, n_base+c]

            for (int e = lane; e < WM * WN; e += 32) {
                int r = e / WN;
                int c = e % WN;
                int j     = m_base + r;
                int v_col = n_base + c;
                float val = wt[e];
                if (j < chunk_len) {
                    dv2_ptr[v_base + (int64_t)j * H * V_DIM + v_col] =
                        __float2bfloat16(val * sm_beta[j]);
                    db_local[r] += val * __bfloat162float(sm_v[j * V_DIM + v_col]);
                }
            }
        }

        // Warp-shuffle reduce db_local[r] across all 32 lanes → atomicAdd to sm_db
        for (int r = 0; r < WM; r++) {
            float v = db_local[r];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v,  8);
            v += __shfl_xor_sync(0xffffffffu, v,  4);
            v += __shfl_xor_sync(0xffffffffu, v,  2);
            v += __shfl_xor_sync(0xffffffffu, v,  1);
            if (lane == 0) atomicAdd(&sm_db[m_base + r], v);
        }
    }
    __syncthreads();

    // ========================================================
    // Phase 4: dkgb = sm_A^T @ sm_dw_neg  [BT×BT]^T × [BT×K] → [BT×K]
    //   Inline: dk[j,k] = dk_inner[j,k] + dkgb[j,k]*gb[j,k]
    //           dg[j,k] = q[j,k]*dq[j,k] - k[j,k]*dk_inner[j,k]
    //                   + m_last[j]*dgk[k] + kg[j,k]*dkgb[j,k]*beta[j]
    //           db[j] += sum_k(dkgb[j,k]*kg[j,k])
    // ========================================================
    {
        const int m_base = warp_id * WM;
        float db_local[WM] = {};

        for (int n_tile = 0; n_tile < K_DIM / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);

            for (int k_tile = 0; k_tile < BT / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_A      + k_base * BT  + m_base, BT);
                wmma::load_matrix_sync(b_frag, sm_dw_neg + k_base * K_DIM + n_base, K_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(wt, c_frag, WN, wmma::mem_row_major);
            // wt[r*WN + c] = dkgb[m_base+r, n_base+c]

            for (int e = lane; e < WM * WN; e += 32) {
                int r    = e / WN;
                int c    = e % WN;
                int j    = m_base + r;
                int k_col = n_base + c;
                float dkgb_val = wt[e];
                if (j < chunk_len) {
                    float g_jk       = g_ptr[((bos + chunk_start + j) * H + i_h) * K_DIM + k_col];
                    float gkexp      = exp2f(g_jk);
                    float gb_jk      = gkexp * sm_beta[j];
                    float kg_jk      = __bfloat162float(sm_kg[j * K_DIM + k_col]);
                    float k_jk       = __bfloat162float(k_ptr[qk_base + (int64_t)j * H * K_DIM + k_col]);
                    float q_jk       = __bfloat162float(q_ptr[qk_base + (int64_t)j * H * K_DIM + k_col]);
                    float dk_inner_jk = sm_dk_inner[j * K_DIM + k_col];
                    float dq_jk      = dq_ptr[qk_base + (int64_t)j * H * K_DIM + k_col];
                    float m_last_j   = (j == last_t) ? 1.f : 0.f;

                    dk_ptr[qk_base + (int64_t)j * H * K_DIM + k_col] =
                        dk_inner_jk + dkgb_val * gb_jk;

                    dg_ptr[((bos + chunk_start + j) * H + i_h) * K_DIM + k_col] =
                        q_jk * dq_jk
                        - k_jk * dk_inner_jk
                        + m_last_j * sm_dgk[k_col]
                        + kg_jk * dkgb_val * sm_beta[j];

                    db_local[r] += dkgb_val * kg_jk;
                }
            }
        }

        for (int r = 0; r < WM; r++) {
            float v = db_local[r];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v,  8);
            v += __shfl_xor_sync(0xffffffffu, v,  4);
            v += __shfl_xor_sync(0xffffffffu, v,  2);
            v += __shfl_xor_sync(0xffffffffu, v,  1);
            if (lane == 0) atomicAdd(&sm_db[m_base + r], v);
        }
    }
    __syncthreads();

    // ========================================================
    // Phase 5: Finalize dA (scalar BT×BT operations) and write db.
    //
    // Triton reference:
    //   m_A = (o_t[:,None] > o_t[None,:]) & (m_t[:,None] & m_t)
    //   b_dA = where(m_A, b_dA * b_beta[None,:], 0)
    //   b_dA = dot(b_dA, b_A)           ← sm_dA @ A_local
    //   b_dA = dot(b_A.T, b_dA)         ← A_local^T @ (prev result)  ... wait
    //
    // Actually the Triton steps are:
    //   b_dA  = where(m_A, b_dA*b_beta[None,:], 0)
    //   b_dA2 = dot(b_dA.to(bf16), b_A)          [BT×BT] = [BT×BT] @ [BT×BT]
    //   b_dA  = dot(b_A.T, b_dA2.to(bf16))       [BT×BT] = [BT×BT]^T @ [BT×BT]
    //   dA    = where(m_A, -b_dA, 0)
    // ========================================================

    // Step 1: Apply lower-triangular mask and beta[j] to sm_dA
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;
        sm_dA[ij] = (i > j && i < chunk_len && j < chunk_len)
            ? sm_dA[ij] * sm_beta[j] : 0.f;
    }
    __syncthreads();

    // Steps 2–3: replace scalar BT×BT GEMMs with WMMA.
    // sm_dv is free at this point (last used in Phase 3). Reuse its space
    // (16384B ≥ BT*BT*2 = 8192B) as a bf16 scratch buffer.
    __nv_bfloat16* sm_bf_scratch = sm_dv;  // OFF_DV, 16384B free

    // Convert sm_dA (fp32) → sm_bf_scratch (bf16)
    for (int idx = tid; idx < BT * BT; idx += NTHREADS)
        sm_bf_scratch[idx] = __float2bfloat16(sm_dA[idx]);
    __syncthreads();

    // WMMA Step 2: sm_dA2 = sm_dA_bf16 @ sm_A^T  [BT×BT]
    //   a_frag row_major: sm_bf_scratch[m_base:+16, k_base:+16]
    //   b_frag col_major: sm_A[n_base:+16, k_base:+16] → acts as sm_A^T ✓
    {
        const int m_base = warp_id * WM;
        for (int n_tile = 0; n_tile < BT / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);
            for (int k_tile = 0; k_tile < BT / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_bf_scratch + m_base * BT + k_base, BT);
                wmma::load_matrix_sync(b_frag, sm_A          + n_base * BT + k_base, BT);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(sm_dA2 + m_base * BT + n_base, c_frag, BT, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Convert sm_dA2 (fp32) → sm_bf_scratch (bf16)
    for (int idx = tid; idx < BT * BT; idx += NTHREADS)
        sm_bf_scratch[idx] = __float2bfloat16(sm_dA2[idx]);
    __syncthreads();

    // WMMA Step 3: sm_dA = sm_A^T @ sm_dA2_bf16  [BT×BT]
    //   a_frag col_major with ptr=sm_A+k_base*BT+m_base:
    //     frag[r,c] = sm_A[(k_base+c)*BT+(m_base+r)] = (sm_A^T)[m_base+r, k_base+c] ✓
    //   b_frag row_major: sm_bf_scratch[k_base:+16, n_base:+16]
    {
        const int m_base = warp_id * WM;
        for (int n_tile = 0; n_tile < BT / WN; n_tile++) {
            const int n_base = n_tile * WN;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);
            for (int k_tile = 0; k_tile < BT / WK; k_tile++) {
                const int k_base = k_tile * WK;
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, sm_A          + k_base * BT + m_base, BT);
                wmma::load_matrix_sync(b_frag, sm_bf_scratch + k_base * BT + n_base, BT);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(sm_dA + m_base * BT + n_base, c_frag, BT, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Step 4: Apply final mask, negate, write to global dA
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;
        if (i < chunk_len) {
            float val = (i > j && j < chunk_len) ? -sm_dA[ij] : 0.f;
            dA_ptr[A_base + (int64_t)i * H * BT + j] = val;
        }
    }

    // Write sm_db to global
    for (int t = tid; t < BT; t += NTHREADS) {
        if (t < chunk_len)
            db_ptr[beta_base + (int64_t)t * H] = sm_db[t];
    }
}

// ---- Host launcher ----
struct KDABwdDqkgParams {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* v_new_ptr;
    const void* g_ptr;
    const void* beta_ptr;
    const void* A_ptr;
    const void* h_ptr;
    const void* do_ptr;
    const void* dh_ptr;
    const void* dv_ptr;
    void*       dq_ptr;
    void*       dk_ptr;
    void*       dv2_ptr;
    void*       dg_ptr;
    void*       db_ptr;
    void*       dA_ptr;
    int B, T, H;
    float scale;
};

void launch_chunk_kda_bwd_dqkg(const KDABwdDqkgParams& p, cudaStream_t stream)
{
    static bool smem_set = false;
    if (!smem_set) {
        cudaFuncSetAttribute(
            chunk_kda_bwd_dqkg_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES);
        smem_set = true;
    }

    const int NT = (p.T + BT - 1) / BT;
    dim3 grid(NT, p.B * p.H);
    dim3 block(NTHREADS);

    chunk_kda_bwd_dqkg_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        (const __nv_bfloat16*)p.q_ptr,
        (const __nv_bfloat16*)p.k_ptr,
        (const __nv_bfloat16*)p.v_ptr,
        (const __nv_bfloat16*)p.v_new_ptr,
        (const float*)         p.g_ptr,
        (const float*)         p.beta_ptr,
        (const __nv_bfloat16*)p.A_ptr,
        (const __nv_bfloat16*)p.h_ptr,
        (const __nv_bfloat16*)p.do_ptr,
        (const __nv_bfloat16*)p.dh_ptr,
        (const __nv_bfloat16*)p.dv_ptr,
        (float*)               p.dq_ptr,
        (float*)               p.dk_ptr,
        (__nv_bfloat16*)       p.dv2_ptr,
        (float*)               p.dg_ptr,
        (float*)               p.db_ptr,
        (float*)               p.dA_ptr,
        p.B, p.T, p.H, p.scale);
}
