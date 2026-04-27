// Copyright 2025-2026 Ant Group Co., Ltd.
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

// Phase 1 SM90 CUDA implementation of chunk_kda_bwd_wy_dqkg_fused.
// Simple single-CTA-per-tile implementation (no WGMMA, no TMA).
// Faithfully translates the FLA Triton kernel math.
// Reference: third_party/flash-linear-attention/fla/ops/kda/chunk_bwd.py
//            chunk_kda_bwd_kernel_wy_dqkg_fused (TRANSPOSE_STATE=False, IS_VARLEN=False)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// ---- Compile-time constants ----
static constexpr int NTHREADS = 128;
static constexpr int BT       = 64;
static constexpr int K_DIM    = 128;
static constexpr int V_DIM    = 128;

// ---- Shared memory layout (bytes) ----
// sm_h[K*V]    bf16  = 32768 B  @ 0
// sm_dh[K*V]   bf16  = 32768 B  @ 32768
// sm_A[BT*BT]  bf16  =  8192 B  @ 65536
// sm_dw[BT*K]  bf16  = 16384 B  @ 73728
// sm_kg[BT*K]  bf16  = 16384 B  @ 90112
// sm_dA[BT*BT] f32   = 16384 B  @ 106496
// sm_dA2[BT*BT]f32   = 16384 B  @ 122880
// sm_dgk[K]    f32   =   512 B  @ 139264
// sm_gn[K]     f32   =   512 B  @ 139776
// sm_db[BT]    f32   =   256 B  @ 140288
// sm_beta[BT]  f32   =   256 B  @ 140544
// Total: 140800 B

static constexpr int OFF_H    = 0;
static constexpr int OFF_DH   = 32768;
static constexpr int OFF_A    = 65536;
static constexpr int OFF_DW   = 73728;
static constexpr int OFF_KG   = 90112;
static constexpr int OFF_DA   = 106496;
static constexpr int OFF_DA2  = 122880;
static constexpr int OFF_DGK  = 139264;
static constexpr int OFF_GN   = 139776;
static constexpr int OFF_DB   = 140288;
static constexpr int OFF_BETA = 140544;
static constexpr int SMEM_BYTES = 140800;

// ---- Kernel ----
// Grid: (NT, B*H)  — one CTA per (chunk, batch-head) pair.
// Block: (NTHREADS=128,) — thread k handles the k-th K-feature dimension.
//
// Inputs:
//   q, k, v, v_new, g, beta, A, h, do_, dh, dv
// Outputs:
//   dq, dk, dv2, dg, db, dA
//
// Tensor layouts (non-varlen, TRANSPOSE_STATE=False):
//   q/k/g/dq/dk/dg:  [B*T, H, K]   row-major
//   v/v_new/do_/dv/dv2: [B*T, H, V] row-major
//   beta/db:          [B*T, H]      row-major
//   A/dA:             [B*T, H, BT]  row-major  (last chunk dim is always BT)
//   h/dh:             [NT*B, H, K, V] row-major (K outer, V inner)
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

    const int NT          = (T + BT - 1) / BT;
    const int64_t i_tg    = (int64_t)i_b * NT + i_t;
    const int chunk_start = i_t * BT;
    const int chunk_len   = (chunk_start + BT <= T) ? BT : (T - chunk_start);
    const int last_t      = chunk_len - 1;

    // Base offsets into [B*T, H, *] tensors for this chunk's first token and head
    const int64_t bos        = (int64_t)i_b * T;
    const int64_t tok0_h     = (bos + chunk_start) * H + i_h;   // (t_abs * H + i_h) at t=0
    const int64_t qk_base    = tok0_h * K_DIM;      // index of q/k/g [tok0, i_h, 0]
    const int64_t v_base     = tok0_h * V_DIM;      // index of v/do  [tok0, i_h, 0]
    const int64_t beta_base  = tok0_h;              // index of beta  [tok0, i_h]
    const int64_t A_base     = tok0_h * BT;         // index of A     [tok0, i_h, 0]
    const int64_t h_base     = (i_tg * H + i_h) * (int64_t)(K_DIM * V_DIM);

    // ---- Shared memory pointers ----
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sm_h    = (__nv_bfloat16*)(smem_raw + OFF_H);
    __nv_bfloat16* sm_dh   = (__nv_bfloat16*)(smem_raw + OFF_DH);
    __nv_bfloat16* sm_A    = (__nv_bfloat16*)(smem_raw + OFF_A);
    __nv_bfloat16* sm_dw   = (__nv_bfloat16*)(smem_raw + OFF_DW);
    __nv_bfloat16* sm_kg   = (__nv_bfloat16*)(smem_raw + OFF_KG);
    float*         sm_dA   = (float*)(smem_raw + OFF_DA);
    float*         sm_dA2  = (float*)(smem_raw + OFF_DA2);
    float*         sm_dgk  = (float*)(smem_raw + OFF_DGK);
    float*         sm_gn   = (float*)(smem_raw + OFF_GN);
    float*         sm_db   = (float*)(smem_raw + OFF_DB);
    float*         sm_beta = (float*)(smem_raw + OFF_BETA);

    // ========================================================
    // Phase 0: Load to shared memory; init accumulators.
    // ========================================================

    // sm_h[k*V+v] = h[i_tg, i_h, k, v]  (row-major [K,V])
    // sm_dh[k*V+v] = dh[i_tg, i_h, k, v]
    for (int idx = tid; idx < K_DIM * V_DIM; idx += NTHREADS) {
        sm_h[idx]  = h_ptr[h_base + idx];
        sm_dh[idx] = dh_ptr[h_base + idx];
    }

    // sm_A[t*BT+j] = A[tok0+t, i_h, j]
    // The j dimension is always BT-wide; only rows t < chunk_len are valid.
    for (int idx = tid; idx < BT * BT; idx += NTHREADS) {
        int t_local = idx / BT;
        int j       = idx % BT;
        if (t_local < chunk_len) {
            sm_A[idx] = A_ptr[A_base + (int64_t)t_local * H * BT + j];
        } else {
            sm_A[idx] = __float2bfloat16(0.f);
        }
    }

    // sm_beta[t] = beta[tok0+t, i_h]
    for (int t = tid; t < BT; t += NTHREADS) {
        sm_beta[t] = (t < chunk_len) ? beta_ptr[beta_base + (int64_t)t * H] : 0.f;
    }

    // sm_gn[k] = g[last valid token in chunk, i_h, k]
    for (int k = tid; k < K_DIM; k += NTHREADS) {
        sm_gn[k] = g_ptr[((bos + chunk_start + last_t) * H + i_h) * K_DIM + k];
    }

    // Init float accumulators to 0
    for (int idx = tid; idx < BT * BT; idx += NTHREADS) {
        sm_dA[idx]  = 0.f;
        sm_dA2[idx] = 0.f;
    }
    for (int k = tid; k < K_DIM; k += NTHREADS) {
        sm_dgk[k] = 0.f;
    }
    for (int t = tid; t < BT; t += NTHREADS) {
        sm_db[t] = 0.f;
    }

    __syncthreads();

    // ========================================================
    // Phase 1: Thread k handles all K-dimension work.
    //   - Computes dgk[k], dq[t,k], dk_inner[t,k], sm_dw[t,k], sm_kg[t,k]
    //   - Writes dq[t,k] to global
    //   - Stores dk_inner in local array (persists across __syncthreads)
    // ========================================================
    const int k = tid;

    // Per-thread local arrays (may spill to per-thread local memory — acceptable)
    float dq_arr[BT];
    float dk_inner_arr[BT];

    // dgk[k] partial: sum_v(h[k,v]*dh[k,v]) — scaled by exp2(gn[k]) below
    float dgk_k = 0.f;
    for (int v = 0; v < V_DIM; v++) {
        dgk_k += __bfloat162float(sm_h[k * V_DIM + v])
               * __bfloat162float(sm_dh[k * V_DIM + v]);
    }

    float gn_k = sm_gn[k];

    for (int t = 0; t < BT; t++) {
        bool valid = (t < chunk_len);

        // Load g[t, k] (log2-domain gate value)
        float g_tk    = valid ? g_ptr[((bos + chunk_start + t) * H + i_h) * K_DIM + k] : 0.f;
        float gkexp   = exp2f(g_tk);

        // Inner products over V dimension
        float dq_raw  = 0.f;
        float dk_raw  = 0.f;
        float dw_raw  = 0.f;

        if (valid) {
            for (int v = 0; v < V_DIM; v++) {
                int64_t vi    = v_base + (int64_t)t * H * V_DIM + v;
                float do_tv   = __bfloat162float(do_ptr[vi]);
                float vnew_tv = __bfloat162float(v_new_ptr[vi]);
                float dv_tv   = __bfloat162float(dv_ptr[vi]);
                float h_kv    = __bfloat162float(sm_h[k * V_DIM + v]);
                float dh_kv   = __bfloat162float(sm_dh[k * V_DIM + v]);
                dq_raw  += do_tv   * h_kv;
                dk_raw  += vnew_tv * dh_kv;
                dw_raw  += dv_tv   * h_kv;
            }
        }

        float dq_final = valid ? dq_raw * gkexp * scale : 0.f;
        // dk_inner = dk_raw * exp2(gn - g) (0 if token invalid)
        float dk_inner = valid ? dk_raw * exp2f(gn_k - g_tk) : 0.f;
        // kg[t, k] = k[t, k] * exp2(g[t, k])
        float k_tk = valid ? __bfloat162float(k_ptr[qk_base + (int64_t)t * H * K_DIM + k]) : 0.f;
        float kg_tk = valid ? k_tk * gkexp : 0.f;

        dq_arr[t]       = dq_final;
        dk_inner_arr[t] = dk_inner;

        // Write dq to global memory
        if (valid) {
            dq_ptr[qk_base + (int64_t)t * H * K_DIM + k] = dq_final;
        }

        // sm_dw[t*K+k] = -dw_raw (negated, as bf16)
        sm_dw[t * K_DIM + k] = __float2bfloat16(valid ? -dw_raw : 0.f);
        // sm_kg[t*K+k] = kg_tk (as bf16)
        sm_kg[t * K_DIM + k] = __float2bfloat16(kg_tk);
    }

    // Finalize dgk[k] = exp2(gn[k]) * sum_v(h*dh) + sum_t(k[t,k] * dk_inner[t,k])
    float sum_kdk = 0.f;
    for (int t = 0; t < chunk_len; t++) {
        float k_tk = __bfloat162float(k_ptr[qk_base + (int64_t)t * H * K_DIM + k]);
        sum_kdk += k_tk * dk_inner_arr[t];
    }
    dgk_k = dgk_k * exp2f(gn_k) + sum_kdk;
    sm_dgk[k] = dgk_k;

    __syncthreads();

    // ========================================================
    // Phase 2: Thread tid computes sm_dA contributions.
    //   sm_dA[i*BT+j] = sum_v(dv[i,v]*v[j,v])  (V contribution)
    //                 + sum_k(sm_dw[i*K+k]*sm_kg[j*K+k])  (K contribution, dw already negated)
    // 128 threads × 32 pairs = 4096 (i,j) entries.
    // ========================================================
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;

        float da_v = 0.f;
        if (i < chunk_len && j < chunk_len) {
            for (int v = 0; v < V_DIM; v++) {
                float dv_iv = __bfloat162float(dv_ptr[v_base + (int64_t)i * H * V_DIM + v]);
                float v_jv  = __bfloat162float(v_ptr[v_base  + (int64_t)j * H * V_DIM + v]);
                da_v += dv_iv * v_jv;
            }
        }

        float da_k = 0.f;
        for (int kk = 0; kk < K_DIM; kk++) {
            // sm_dw[i*K+kk] = -dw[i,kk]; sm_kg[j*K+kk] = kg[j,kk]
            da_k += __bfloat162float(sm_dw[i * K_DIM + kk])
                  * __bfloat162float(sm_kg[j * K_DIM + kk]);
        }

        sm_dA[ij] = da_v + da_k;
    }

    // ========================================================
    // Phase 3: Thread tid computes dvb, dv2, and sm_db (dvb contribution).
    //   dvb[j, v] = sum_s(sm_A[s*BT+j] * dv[s, v])  where sm_A[s*BT+j] = A_local[s,j]
    //   dv2[j, v] = dvb[j, v] * beta[j]
    //   sm_db[j] += sum_v(dvb[j,v] * v[j,v])   (via atomicAdd)
    // 128 threads × 64 pairs = 8192 (j,v) entries.
    // ========================================================
    for (int jv = tid; jv < BT * V_DIM; jv += NTHREADS) {
        int j = jv / V_DIM;
        int v = jv % V_DIM;

        if (j < chunk_len) {
            float dvb = 0.f;
            for (int s = 0; s < chunk_len; s++) {
                float a_sj  = __bfloat162float(sm_A[(int64_t)s * BT + j]);
                float dv_sv = __bfloat162float(dv_ptr[v_base + (int64_t)s * H * V_DIM + v]);
                dvb += a_sj * dv_sv;
            }

            dv2_ptr[v_base + (int64_t)j * H * V_DIM + v] = __float2bfloat16(dvb * sm_beta[j]);

            float v_jv = __bfloat162float(v_ptr[v_base + (int64_t)j * H * V_DIM + v]);
            atomicAdd(&sm_db[j], dvb * v_jv);
        }
    }

    __syncthreads();

    // ========================================================
    // Phase 4: Thread k computes dkgb, finalizes dk and dg, and updates sm_db.
    //   dkgb[j, k] = sum_s(sm_A[s*BT+j] * sm_dw[s*K+k])
    //                (= sum_s A_local^T[j,s] * (-dw[s,k]))
    //   dk[j, k]   = dk_inner[j, k] + dkgb[j, k] * gb[j, k]
    //              where gb[j, k] = exp2(g[j,k]) * beta[j]
    //   dg[j, k]   = q[j,k]*dq[j,k] - k[j,k]*dk_inner[j,k]
    //              + m_last[j]*dgk[k] + kg[j,k]*dkgb[j,k]*beta[j]
    //   sm_db[j]  += sum_k(dkgb[j,k] * kg[j,k])  (via atomicAdd)
    // ========================================================
    for (int j = 0; j < BT; j++) {
        if (j >= chunk_len) continue;

        float dkgb_jk = 0.f;
        for (int s = 0; s < chunk_len; s++) {
            dkgb_jk += __bfloat162float(sm_A[(int64_t)s * BT + j])
                     * __bfloat162float(sm_dw[(int64_t)s * K_DIM + k]);
        }

        float g_jk   = g_ptr[((bos + chunk_start + j) * H + i_h) * K_DIM + k];
        float gkexp  = exp2f(g_jk);
        float gb_jk  = gkexp * sm_beta[j];
        float kg_jk  = __bfloat162float(sm_kg[(int64_t)j * K_DIM + k]);  // k[j,k]*gkexp
        float k_jk   = __bfloat162float(k_ptr[qk_base + (int64_t)j * H * K_DIM + k]);
        float q_jk   = __bfloat162float(q_ptr[qk_base + (int64_t)j * H * K_DIM + k]);

        dk_ptr[qk_base + (int64_t)j * H * K_DIM + k] =
            dk_inner_arr[j] + dkgb_jk * gb_jk;

        float m_last_j = (j == last_t) ? 1.f : 0.f;
        dg_ptr[((bos + chunk_start + j) * H + i_h) * K_DIM + k] =
            q_jk  * dq_arr[j]
            - k_jk * dk_inner_arr[j]
            + m_last_j * sm_dgk[k]
            + kg_jk * dkgb_jk * sm_beta[j];

        atomicAdd(&sm_db[j], dkgb_jk * kg_jk);
    }

    __syncthreads();

    // ========================================================
    // Phase 5: Finalize dA and write db to global.
    //
    // Triton reference:
    //   m_A = (o_t[:,None] > o_t[None,:]) & (m_t[:,None] & m_t)
    //   b_dA = where(m_A, b_dA * b_beta[None,:], 0)
    //   b_dA = dot(b_dA.to(b_A.dtype), b_A)       ← sm_dA @ A_local^T
    //   b_dA = dot(b_A, b_dA.to(b_A.dtype))       ← A_local^T @ (prev result)
    //   b_dA = where(m_A, -b_dA, 0)
    //
    // Step 1: Apply mask and beta[j] to sm_dA.
    // ========================================================
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;
        if (i > j && i < chunk_len && j < chunk_len) {
            sm_dA[ij] = sm_dA[ij] * sm_beta[j];
        } else {
            sm_dA[ij] = 0.f;
        }
    }

    __syncthreads();

    // Step 2: sm_dA2 = sm_dA @ A_local^T
    //   sm_dA2[i,j] = sum_kk(sm_dA[i,kk] * A_local[j,kk])
    //               = sum_kk(sm_dA[i*BT+kk] * sm_A[j*BT+kk])
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;
        float acc = 0.f;
        for (int kk = 0; kk < BT; kk++) {
            acc += sm_dA[i * BT + kk] * __bfloat162float(sm_A[j * BT + kk]);
        }
        sm_dA2[ij] = acc;
    }

    __syncthreads();

    // Step 3: sm_dA = A_local^T @ sm_dA2
    //   sm_dA[i,j] = sum_kk(A_local[kk,i] * sm_dA2[kk,j])
    //              = sum_kk(sm_A[kk*BT+i] * sm_dA2[kk*BT+j])
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;
        float acc = 0.f;
        for (int kk = 0; kk < BT; kk++) {
            acc += __bfloat162float(sm_A[kk * BT + i]) * sm_dA2[kk * BT + j];
        }
        sm_dA[ij] = acc;
    }

    __syncthreads();

    // Step 4: Apply final mask, negate, and write to global dA.
    //   dA[tok0+i, i_h, j] = (i > j && i < chunk_len && j < chunk_len) ? -sm_dA[i,j] : 0
    for (int ij = tid; ij < BT * BT; ij += NTHREADS) {
        int i = ij / BT;
        int j = ij % BT;
        if (i < chunk_len) {
            float val = (i > j && j < chunk_len) ? -sm_dA[ij] : 0.f;
            dA_ptr[A_base + (int64_t)i * H * BT + j] = val;
        }
    }

    // Write sm_db to global db
    for (int t = tid; t < BT; t += NTHREADS) {
        if (t < chunk_len) {
            db_ptr[beta_base + (int64_t)t * H] = sm_db[t];
        }
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
