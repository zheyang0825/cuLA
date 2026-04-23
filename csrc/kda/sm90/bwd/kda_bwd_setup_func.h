#pragma once

// Setup / epilogue helpers ported from the SM100 implementation
// (kda-backward-internal-feat-bwd_intra_opt/csrc/kda_bwd/util_func.h).
// All TMEM-touching operations have been replaced with SMEM access:
// the MMA warpgroup on SM90 produces dQ / dQ2 / dKt into shared memory
// (smem_mma_dq, smem_mma_dq2, smem_mma_dkt) instead of TMEM.

#include <cute/tensor.hpp>
#include <cuda_bf16.h>

#include "kda_bwd_basic.h"
#include "kda_bwd_helpers.h"

namespace sm90_bwd {

using namespace cute;

// ============================================================
// Compute-side functions  (B-operand setup)
// ============================================================

// KG[j,d] = K[j,d] * exp2(G_norm[d] - G[j,d])
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void
setup_kg_intra(
    G_TENSOR& sG, K_TENSOR& sK, KG_TENSOR& sKG_intra,
    int tile_j, int idx_in_warpgroup, float4& gn, int index) {
    int x = idx_in_warpgroup / 8 + tile_j * 16;
    int y = idx_in_warpgroup % 8 * 4;
    float4 tmp = *reinterpret_cast<float4*>(&sG(x, y));
    nvbf16x4 tmp_k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
    float4 res;
    float2 sub1 = float2_sub(reinterpret_cast<float2*>(&gn)[0], reinterpret_cast<float2*>(&tmp)[0]);
    float2 sub2 = float2_sub(reinterpret_cast<float2*>(&gn)[1], reinterpret_cast<float2*>(&tmp)[1]);
    sub1.x = exp2f(sub1.x); sub1.y = exp2f(sub1.y);
    sub2.x = exp2f(sub2.x); sub2.y = exp2f(sub2.y);
    reinterpret_cast<float2*>(&res)[0] = float2_mul(sub1, __bfloat1622float2(tmp_k.a));
    reinterpret_cast<float2*>(&res)[1] = float2_mul(sub2, __bfloat1622float2(tmp_k.b));
    store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index, res);
}

// 2 KG outputs from one (sG, sK) load (same row, 2 G_norm refs)
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void
setup_kg_intra_2gn(
    G_TENSOR& sG, K_TENSOR& sK, KG_TENSOR& sKG_intra,
    int tile_j, int idx_in_warpgroup,
    float4& gn1, float4& gn2, int index1, int index2) {
    int x = idx_in_warpgroup / 8 + tile_j * 16;
    int y = idx_in_warpgroup % 8 * 4;
    float4 g = *reinterpret_cast<float4*>(&sG(x, y));
    nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
    float2 kf_a = __bfloat1622float2(k.a);
    float2 kf_b = __bfloat1622float2(k.b);
    float2 s1a = float2_sub(reinterpret_cast<float2*>(&gn1)[0], reinterpret_cast<float2*>(&g)[0]);
    float2 s1b = float2_sub(reinterpret_cast<float2*>(&gn1)[1], reinterpret_cast<float2*>(&g)[1]);
    s1a.x = exp2f(s1a.x); s1a.y = exp2f(s1a.y);
    s1b.x = exp2f(s1b.x); s1b.y = exp2f(s1b.y);
    float4 res1;
    reinterpret_cast<float2*>(&res1)[0] = float2_mul(s1a, kf_a);
    reinterpret_cast<float2*>(&res1)[1] = float2_mul(s1b, kf_b);
    store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index1, res1);
    float2 s2a = float2_sub(reinterpret_cast<float2*>(&gn2)[0], reinterpret_cast<float2*>(&g)[0]);
    float2 s2b = float2_sub(reinterpret_cast<float2*>(&gn2)[1], reinterpret_cast<float2*>(&g)[1]);
    s2a.x = exp2f(s2a.x); s2a.y = exp2f(s2a.y);
    s2b.x = exp2f(s2b.x); s2b.y = exp2f(s2b.y);
    float4 res2;
    reinterpret_cast<float2*>(&res2)[0] = float2_mul(s2a, kf_a);
    reinterpret_cast<float2*>(&res2)[1] = float2_mul(s2b, kf_b);
    store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index2, res2);
}

// QG[x,d]=Q*exp2(G-Gn), KBG[x,d]=K*exp2(G-Gn)*beta[x]
template <typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR, typename QKG_TENSOR, int QKG_OFFSET>
__forceinline__ __device__ void
setup_qkg_intra(
    G_TENSOR& sG, Q_TENSOR& sQ, K_TENSOR& sK, QKG_TENSOR& sQKG_intra,
    int tile_j, int idx_in_warpgroup, int sub_seq_len,
    float2& beta, float4& gn, int index) {
    int x = idx_in_warpgroup / 8 + tile_j * 16;
    int y = idx_in_warpgroup % 8 * 4;
    if (x < sub_seq_len) {
        float4 tmp = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 tmp_k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        nvbf16x4 tmp_q = *reinterpret_cast<nvbf16x4*>(&sQ(x, y));
        float4 res_q, res_k;
        float2 sub1 = float2_sub(reinterpret_cast<float2*>(&tmp)[0], reinterpret_cast<float2*>(&gn)[0]);
        float2 sub2 = float2_sub(reinterpret_cast<float2*>(&tmp)[1], reinterpret_cast<float2*>(&gn)[1]);
        sub1.x = exp2f(sub1.x); sub1.y = exp2f(sub1.y);
        sub2.x = exp2f(sub2.x); sub2.y = exp2f(sub2.y);
        reinterpret_cast<float2*>(&res_q)[0] = float2_mul(sub1, __bfloat1622float2(tmp_q.a));
        reinterpret_cast<float2*>(&res_q)[1] = float2_mul(sub2, __bfloat1622float2(tmp_q.b));
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * index, res_q);
        sub1 = float2_mul(sub1, __bfloat1622float2(tmp_k.a));
        sub2 = float2_mul(sub2, __bfloat1622float2(tmp_k.b));
        reinterpret_cast<float2*>(&res_k)[0] = float2_mul(sub1, beta);
        reinterpret_cast<float2*>(&res_k)[1] = float2_mul(sub2, beta);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * index, res_k);
    } else {
        float4 res_zero = {0.0f, 0.0f, 0.0f, 0.0f};
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * index, res_zero);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * index, res_zero);
    }
}

// 2×(QG, KBG) outputs sharing the same row's sG/sK/sQ load.
template <typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR, typename QKG_TENSOR, int QKG_OFFSET>
__forceinline__ __device__ void
setup_qkg_intra_2gn(
    G_TENSOR& sG, Q_TENSOR& sQ, K_TENSOR& sK, QKG_TENSOR& sQKG_intra,
    int tile_j, int idx_in_warpgroup, int sub_seq_len,
    float2& beta, float4& gn1, float4& gn2, int index1, int index2) {
    int x = idx_in_warpgroup / 8 + tile_j * 16;
    int y = idx_in_warpgroup % 8 * 4;
    if (x < sub_seq_len) {
        float4 g = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        nvbf16x4 q = *reinterpret_cast<nvbf16x4*>(&sQ(x, y));
        float2 qf_a = __bfloat1622float2(q.a), qf_b = __bfloat1622float2(q.b);
        float2 kf_a = __bfloat1622float2(k.a), kf_b = __bfloat1622float2(k.b);
        float2 s1a = float2_sub(reinterpret_cast<float2*>(&g)[0], reinterpret_cast<float2*>(&gn1)[0]);
        float2 s1b = float2_sub(reinterpret_cast<float2*>(&g)[1], reinterpret_cast<float2*>(&gn1)[1]);
        s1a.x = exp2f(s1a.x); s1a.y = exp2f(s1a.y);
        s1b.x = exp2f(s1b.x); s1b.y = exp2f(s1b.y);
        float4 rq1; reinterpret_cast<float2*>(&rq1)[0] = float2_mul(s1a, qf_a); reinterpret_cast<float2*>(&rq1)[1] = float2_mul(s1b, qf_b);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * index1, rq1);
        float4 rk1;
        reinterpret_cast<float2*>(&rk1)[0] = float2_mul(float2_mul(s1a, kf_a), beta);
        reinterpret_cast<float2*>(&rk1)[1] = float2_mul(float2_mul(s1b, kf_b), beta);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * index1, rk1);
        float2 s2a = float2_sub(reinterpret_cast<float2*>(&g)[0], reinterpret_cast<float2*>(&gn2)[0]);
        float2 s2b = float2_sub(reinterpret_cast<float2*>(&g)[1], reinterpret_cast<float2*>(&gn2)[1]);
        s2a.x = exp2f(s2a.x); s2a.y = exp2f(s2a.y);
        s2b.x = exp2f(s2b.x); s2b.y = exp2f(s2b.y);
        float4 rq2; reinterpret_cast<float2*>(&rq2)[0] = float2_mul(s2a, qf_a); reinterpret_cast<float2*>(&rq2)[1] = float2_mul(s2b, qf_b);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * index2, rq2);
        float4 rk2;
        reinterpret_cast<float2*>(&rk2)[0] = float2_mul(float2_mul(s2a, kf_a), beta);
        reinterpret_cast<float2*>(&rk2)[1] = float2_mul(float2_mul(s2b, kf_b), beta);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * index2, rk2);
    } else {
        float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * index1, z);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * index1, z);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * index2, z);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * index2, z);
    }
}

// Fused KG + QG + KBG from one row's sG/sK/sQ load (2 distinct gn's).
template <typename G_TENSOR, typename K_TENSOR, typename Q_TENSOR,
          typename KG_TENSOR, typename QKG_TENSOR, int KG_OFFSET, int QKG_OFFSET>
__forceinline__ __device__ void
setup_intra_fused(
    G_TENSOR& sG, K_TENSOR& sK, Q_TENSOR& sQ,
    KG_TENSOR& sKG_intra, QKG_TENSOR& sQKG_intra,
    int tile_j, int idx_in_warpgroup, int sub_seq_len,
    float4& gn_kg, float4& gn_qkg, float2& beta,
    int kg_index, int qkg_index) {
    int x = idx_in_warpgroup / 8 + tile_j * 16;
    int y = idx_in_warpgroup % 8 * 4;
    if (x < sub_seq_len) {
        float4 g = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        nvbf16x4 q = *reinterpret_cast<nvbf16x4*>(&sQ(x, y));
        float2 kf_a = __bfloat1622float2(k.a);
        float2 kf_b = __bfloat1622float2(k.b);
        float2 ska = float2_sub(reinterpret_cast<float2*>(&gn_kg)[0], reinterpret_cast<float2*>(&g)[0]);
        float2 skb = float2_sub(reinterpret_cast<float2*>(&gn_kg)[1], reinterpret_cast<float2*>(&g)[1]);
        ska.x = exp2f(ska.x); ska.y = exp2f(ska.y);
        skb.x = exp2f(skb.x); skb.y = exp2f(skb.y);
        float4 kg_res;
        reinterpret_cast<float2*>(&kg_res)[0] = float2_mul(ska, kf_a);
        reinterpret_cast<float2*>(&kg_res)[1] = float2_mul(skb, kf_b);
        store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * kg_index, kg_res);
        float2 sqa = float2_sub(reinterpret_cast<float2*>(&g)[0], reinterpret_cast<float2*>(&gn_qkg)[0]);
        float2 sqb = float2_sub(reinterpret_cast<float2*>(&g)[1], reinterpret_cast<float2*>(&gn_qkg)[1]);
        sqa.x = exp2f(sqa.x); sqa.y = exp2f(sqa.y);
        sqb.x = exp2f(sqb.x); sqb.y = exp2f(sqb.y);
        float4 qkg_q_res;
        reinterpret_cast<float2*>(&qkg_q_res)[0] = float2_mul(sqa, __bfloat1622float2(q.a));
        reinterpret_cast<float2*>(&qkg_q_res)[1] = float2_mul(sqb, __bfloat1622float2(q.b));
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * qkg_index, qkg_q_res);
        float4 qkg_k_res;
        sqa = float2_mul(sqa, kf_a);
        sqb = float2_mul(sqb, kf_b);
        reinterpret_cast<float2*>(&qkg_k_res)[0] = float2_mul(sqa, beta);
        reinterpret_cast<float2*>(&qkg_k_res)[1] = float2_mul(sqb, beta);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * qkg_index, qkg_k_res);
    } else {
        float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
        store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * kg_index, z);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8) + QKG_OFFSET * qkg_index, z);
        store_128b(&sQKG_intra(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * qkg_index, z);
    }
}

// Fused inter: KG_inter (neg_exp * k), QG_inter (exp * q), KBG_inter (exp * k * beta).
template <typename G_TENSOR, typename K_TENSOR, typename Q_TENSOR,
          typename KG_TENSOR, typename QKG_TENSOR, int KG_OFFSET, int QKG_OFFSET>
__forceinline__ __device__ void
setup_inter_fused(
    G_TENSOR& sG, K_TENSOR& sK, Q_TENSOR& sQ,
    KG_TENSOR& sKG_inter, QKG_TENSOR& sQKG_inter,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    float2& beta, float4& gn_half) {
    int y = idx_in_warpgroup % 8 * 4;
    gn_half = *reinterpret_cast<float4*>(&sG(min(sub_tile_i * 16 + 8, sub_seq_len - 1), y));
    int x = idx_in_warpgroup / 8 + sub_tile_i * 16;
    if (x < sub_seq_len) {
        float4 tmp = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 tmp_k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        nvbf16x4 tmp_q = *reinterpret_cast<nvbf16x4*>(&sQ(x, y));
        float2 sub1 = float2_sub(reinterpret_cast<float2*>(&tmp)[0], reinterpret_cast<float2*>(&gn_half)[0]);
        float2 sub2 = float2_sub(reinterpret_cast<float2*>(&tmp)[1], reinterpret_cast<float2*>(&gn_half)[1]);
        float4 res_exp, res_neg_exp;
        res_exp.x = exp2f(sub1.x); res_exp.y = exp2f(sub1.y);
        res_exp.z = exp2f(sub2.x); res_exp.w = exp2f(sub2.y);
        res_neg_exp.x = exp2f(-sub1.x); res_neg_exp.y = exp2f(-sub1.y);
        res_neg_exp.z = exp2f(-sub2.x); res_neg_exp.w = exp2f(-sub2.y);
        float4 res_kg;
        reinterpret_cast<float2*>(&res_kg)[0] = float2_mul(reinterpret_cast<float2*>(&res_neg_exp)[0], __bfloat1622float2(tmp_k.a));
        reinterpret_cast<float2*>(&res_kg)[1] = float2_mul(reinterpret_cast<float2*>(&res_neg_exp)[1], __bfloat1622float2(tmp_k.b));
        store_128b(&sKG_inter(y, idx_in_warpgroup / 8) + KG_OFFSET * sub_tile_i, res_kg);
        float4 res_q;
        reinterpret_cast<float2*>(&res_q)[0] = float2_mul(__bfloat1622float2(tmp_q.a), reinterpret_cast<float2*>(&res_exp)[0]);
        reinterpret_cast<float2*>(&res_q)[1] = float2_mul(__bfloat1622float2(tmp_q.b), reinterpret_cast<float2*>(&res_exp)[1]);
        store_128b(&sQKG_inter(y, idx_in_warpgroup / 8) + QKG_OFFSET * sub_tile_i, res_q);
        float4 res_kbeta;
        reinterpret_cast<float2*>(&res_kbeta)[0] = float2_mul(__bfloat1622float2(tmp_k.a), reinterpret_cast<float2*>(&res_exp)[0]);
        reinterpret_cast<float2*>(&res_kbeta)[1] = float2_mul(__bfloat1622float2(tmp_k.b), reinterpret_cast<float2*>(&res_exp)[1]);
        reinterpret_cast<float2*>(&res_kbeta)[0] = float2_mul(reinterpret_cast<float2*>(&res_kbeta)[0], beta);
        reinterpret_cast<float2*>(&res_kbeta)[1] = float2_mul(reinterpret_cast<float2*>(&res_kbeta)[1], beta);
        store_128b(&sQKG_inter(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * sub_tile_i, res_kbeta);
    } else {
        float4 res_zero = {0.0f, 0.0f, 0.0f, 0.0f};
        store_128b(&sKG_inter(y, idx_in_warpgroup / 8) + KG_OFFSET * sub_tile_i, res_zero);
        store_128b(&sQKG_inter(y, idx_in_warpgroup / 8) + QKG_OFFSET * sub_tile_i, res_zero);
        store_128b(&sQKG_inter(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * sub_tile_i, res_zero);
    }
}

// ============================================================
// Epilogue-side functions (TMEM replaced with SMEM)
// ============================================================

// scale[i] = exp2(g[row] - g[block_start])  for i in [K_OFFSET, K_OFFSET+K_TILE)
template <int K_TILE_, int K_OFFSET = 0, typename G_TENSOR>
__forceinline__ __device__ void
epilogue_compute_intra_scale(
    G_TENSOR& sG, int idx_in_warpgroup, float* scale) {
    if (idx_in_warpgroup % 64 >= 16) {
        for (int i = 0; i < K_TILE_ / 4; ++i) {
            float4 bgn = *reinterpret_cast<float4*>(&sG((idx_in_warpgroup % 64) / 16 * 16, K_OFFSET + i * 4));
            float4 bg  = *reinterpret_cast<float4*>(&sG(idx_in_warpgroup % 64, K_OFFSET + i * 4));
            float2 d0 = float2_sub(reinterpret_cast<float2*>(&bg)[0], reinterpret_cast<float2*>(&bgn)[0]);
            float2 d1 = float2_sub(reinterpret_cast<float2*>(&bg)[1], reinterpret_cast<float2*>(&bgn)[1]);
            scale[i * 4]     = exp2f(d0.x);
            scale[i * 4 + 1] = exp2f(d0.y);
            scale[i * 4 + 2] = exp2f(d1.x);
            scale[i * 4 + 3] = exp2f(d1.y);
        }
    }
}

// SM90 layout for MMA-produced dQ / dQ2 / dKt scratch (fp32, [T_TILE, K_TILE]).
// Each of the 256 prep threads owns one (row, K_OFFSET..) slice; row = idx%64.
// Read directly from SMEM.
template <int K_TILE_, int K_OFFSET = 0, typename DQ_SMEM_TENSOR>
__forceinline__ __device__ void
epilogue_apply_dq_intra_smem(
    DQ_SMEM_TENSOR& sMmaDQ, int idx_in_warpgroup, float* res, float* scale) {
    int row = idx_in_warpgroup % 64;
    for (int i = 0; i < K_TILE_ / 4; ++i) {
        *reinterpret_cast<float4*>(&res[i * 4]) = *reinterpret_cast<float4*>(&sMmaDQ(row, K_OFFSET + i * 4));
    }
    if (idx_in_warpgroup % 64 >= 16) {
        for (int i = 0; i < K_TILE_ / 2; ++i) {
            reinterpret_cast<float2*>(res)[i] = float2_mul(reinterpret_cast<float2*>(res)[i], reinterpret_cast<float2*>(scale)[i]);
        }
    } else {
        for (int i = 0; i < K_TILE_ / 2; ++i) {
            reinterpret_cast<float2*>(res)[i] = {0.0f, 0.0f};
        }
    }
}

// res += dq2 * scale (dq2 from SMEM)
template <int K_TILE_, int K_OFFSET = 0, typename DQ2_SMEM_TENSOR>
__forceinline__ __device__ void
epilogue_combine_dq_inter_smem(
    DQ2_SMEM_TENSOR& sMmaDQ2, int idx_in_warpgroup,
    float* res, float* scale) {
    int row = idx_in_warpgroup % 64;
    float res2[K_TILE_];
    for (int i = 0; i < K_TILE_ / 4; ++i) {
        *reinterpret_cast<float4*>(&res2[i * 4]) = *reinterpret_cast<float4*>(&sMmaDQ2(row, K_OFFSET + i * 4));
    }
    for (int i = 0; i < K_TILE_ / 2; ++i) {
        reinterpret_cast<float2*>(res)[i] = float2_fma(
            reinterpret_cast<float2*>(&res2)[i],
            reinterpret_cast<float2*>(scale)[i],
            reinterpret_cast<float2*>(res)[i]);
    }
}

// Output dq result (lower half), also turns res into res*q for dg prep.
template <int K_TILE_, int K_OFFSET = 0, typename Q_TENSOR, typename DQ_TENSOR>
__forceinline__ __device__ void
epilogue_output_dq(
    Q_TENSOR& sQ, DQ_TENSOR& sDQ,
    int idx_in_warpgroup, int sub_seq_len,
    float* res, float* dq_out_base) {
    if (idx_in_warpgroup % 64 < sub_seq_len) {
        int row = idx_in_warpgroup % 64;
        for (int i = 0; i < K_TILE_ / 16; ++i) {
            float dq_res_fp32[16];
            for (int j = 0; j < 4; ++j) {
                float4 tmp_dq = *reinterpret_cast<float4*>(&sDQ(row, K_OFFSET + i * 16 + j * 4));
                for (int k = 0; k < 2; ++k) {
                    reinterpret_cast<float2*>(&tmp_dq)[k] = float2_add(
                        reinterpret_cast<float2*>(&tmp_dq)[k],
                        reinterpret_cast<float2*>(&res[i * 16])[j * 2 + k]);
                }
                *reinterpret_cast<float4*>(&dq_res_fp32[j * 4]) = tmp_dq;
                nvbf16x4 tmp_q = *reinterpret_cast<nvbf16x4*>(&sQ(row, K_OFFSET + i * 16 + j * 4));
                reinterpret_cast<float2*>(res)[i * 8 + j * 2] =
                    float2_mul(reinterpret_cast<float2*>(res)[i * 8 + j * 2], __bfloat1622float2(tmp_q.a));
                reinterpret_cast<float2*>(res)[i * 8 + j * 2 + 1] =
                    float2_mul(reinterpret_cast<float2*>(res)[i * 8 + j * 2 + 1], __bfloat1622float2(tmp_q.b));
            }
            // Write 16 fp32 (64 B) via two 32-B stores.
            store_256B(&dq_res_fp32[0], dq_out_base + i * 16);
            store_256B(&dq_res_fp32[8], dq_out_base + i * 16 + 8);
        }
    }
}

// db accumulation and beta scaling (upper half).
template <int K_TILE_, int K_OFFSET = 0, typename K_TENSOR>
__forceinline__ __device__ void
epilogue_accumulate_db(
    K_TENSOR& sK, int idx_in_warpgroup, int sub_seq_len,
    float* res, float& db, bool is_last_k,
    float* db_out_addr, bf16 beta_val) {
    int row = idx_in_warpgroup % 64;
    for (int i = 0; i < K_TILE_ / 4; ++i) {
        bf16x4 tmp_k = *reinterpret_cast<bf16x4*>(&sK(row, K_OFFSET + i * 4));
        db += float(tmp_k.a) * res[i * 4];
        db += float(tmp_k.b) * res[i * 4 + 1];
        db += float(tmp_k.c) * res[i * 4 + 2];
        db += float(tmp_k.d) * res[i * 4 + 3];
    }
    if (is_last_k && row < sub_seq_len && db_out_addr) {
        *db_out_addr = db;
    }
    float2 beta2 = __bfloat1622float2(__bfloat162bfloat162((__nv_bfloat16)beta_val));
    for (int i = 0; i < K_TILE_ / 2; ++i) {
        reinterpret_cast<float2*>(res)[i] = float2_mul(reinterpret_cast<float2*>(res)[i], beta2);
    }
}

// scale for dkt processing (computed directly from sG)
template <int K_TILE_, int K_OFFSET = 0, typename G_TENSOR>
__forceinline__ __device__ void
epilogue_compute_dkt_scale(
    G_TENSOR& sG, int idx_in_warpgroup, int sub_seq_len, float* scale) {
    int row = idx_in_warpgroup % 64;
    int ref_row = (idx_in_warpgroup < 64)
        ? min((row / 16 + 1) * 16, sub_seq_len - 1)
        : min(row / 16 * 16 + 8, sub_seq_len - 1);
    for (int i = 0; i < K_TILE_ / 4; ++i) {
        float4 bg_ref = *reinterpret_cast<float4*>(&sG(ref_row, K_OFFSET + i * 4));
        float4 bg     = *reinterpret_cast<float4*>(&sG(row,     K_OFFSET + i * 4));
        float2 d0 = float2_sub(reinterpret_cast<float2*>(&bg_ref)[0], reinterpret_cast<float2*>(&bg)[0]);
        float2 d1 = float2_sub(reinterpret_cast<float2*>(&bg_ref)[1], reinterpret_cast<float2*>(&bg)[1]);
        scale[i * 4]     = exp2f(d0.x);
        scale[i * 4 + 1] = exp2f(d0.y);
        scale[i * 4 + 2] = exp2f(d1.x);
        scale[i * 4 + 3] = exp2f(d1.y);
    }
}

// Read MMA dKt result from SMEM and apply scale; zero invalid rows.
template <int K_TILE_, int K_OFFSET = 0, typename DKT_SMEM_TENSOR>
__forceinline__ __device__ void
epilogue_process_dkt_smem(
    DKT_SMEM_TENSOR& sMmaDKt, int idx_in_warpgroup,
    float* res_dkt, float* scale, int sub_seq_len) {
    int row = idx_in_warpgroup % 64;
    for (int i = 0; i < K_TILE_ / 4; ++i) {
        *reinterpret_cast<float4*>(&res_dkt[i * 4]) = *reinterpret_cast<float4*>(&sMmaDKt(row, K_OFFSET + i * 4));
    }
    bool should_zero = (idx_in_warpgroup < 64)
        ? ((row / 16 + 1) * 16 >= sub_seq_len)
        : (row >= sub_seq_len);
    if (should_zero) {
        for (int i = 0; i < K_TILE_ / 2; ++i) {
            reinterpret_cast<float2*>(res_dkt)[i] = {0.0f, 0.0f};
        }
    } else {
        for (int i = 0; i < K_TILE_ / 2; ++i) {
            reinterpret_cast<float2*>(res_dkt)[i] = float2_mul(
                reinterpret_cast<float2*>(res_dkt)[i],
                reinterpret_cast<float2*>(scale)[i]);
        }
    }
}

// Exchange dkt data via smem between lower (<64) and upper (>=64) halves.
// lower writes res_dkt -> sDKT_0; upper writes (res-res_dkt) -> sDKT_1.
template <int K_TILE_, int K_OFFSET = 0, typename DKT_TENSOR>
__forceinline__ __device__ void
epilogue_exchange_dkt(
    DKT_TENSOR& sDKT_0, DKT_TENSOR& sDKT_1,
    int idx_in_warpgroup, float* res, float* res_dkt) {
    if (idx_in_warpgroup < 64) {
        for (int i = 0; i < K_TILE_ / 4; ++i) {
            store_128b(&sDKT_0(idx_in_warpgroup % 64, K_OFFSET + i * 4),
                       *reinterpret_cast<float4*>(&res_dkt[i * 4]));
        }
    } else {
        for (int i = 0; i < K_TILE_ / 4; ++i) {
            float4 dk_sub_dkt;
            reinterpret_cast<float2*>(&dk_sub_dkt)[0] = float2_sub(
                reinterpret_cast<float2*>(res)[i * 2], reinterpret_cast<float2*>(res_dkt)[i * 2]);
            reinterpret_cast<float2*>(&dk_sub_dkt)[1] = float2_sub(
                reinterpret_cast<float2*>(res)[i * 2 + 1], reinterpret_cast<float2*>(res_dkt)[i * 2 + 1]);
            store_128b(&sDKT_1(idx_in_warpgroup % 64, K_OFFSET + i * 4), dk_sub_dkt);
        }
    }
}

// Output dg = res + (dk_sub_dkt - res_dkt) * k + dg_inter   (lower half).
template <int K_TILE_, int K_OFFSET = 0, typename K_TENSOR, typename DG_TENSOR, typename DKT_TENSOR>
__forceinline__ __device__ void
epilogue_output_dg(
    K_TENSOR& sK, DG_TENSOR& sDG, DKT_TENSOR& sDKT_1,
    int idx_in_warpgroup,
    float* res, float* res_dkt, float* dg_out_base) {
    int row = idx_in_warpgroup % 64;
    for (int i = 0; i < K_TILE_ / 4; ++i) {
        float4 dk_sub_dkt = *reinterpret_cast<float4*>(&sDKT_1(row, K_OFFSET + i * 4));
        nvbf16x4 tmp_k = *reinterpret_cast<nvbf16x4*>(&sK(row, K_OFFSET + i * 4));
        float4 dg = *reinterpret_cast<float4*>(&sDG(row, K_OFFSET + i * 4));
        float2 k0 = __bfloat1622float2(tmp_k.a);
        float2 k1 = __bfloat1622float2(tmp_k.b);
        float2 diff0 = float2_sub(reinterpret_cast<float2*>(&dk_sub_dkt)[0], reinterpret_cast<float2*>(res_dkt)[i * 2]);
        float2 diff1 = float2_sub(reinterpret_cast<float2*>(&dk_sub_dkt)[1], reinterpret_cast<float2*>(res_dkt)[i * 2 + 1]);
        float2 res_dg0 = float2_add(reinterpret_cast<float2*>(res)[i * 2],     reinterpret_cast<float2*>(&dg)[0]);
        float2 res_dg1 = float2_add(reinterpret_cast<float2*>(res)[i * 2 + 1], reinterpret_cast<float2*>(&dg)[1]);
        reinterpret_cast<float2*>(res)[i * 2]     = float2_fma(diff0, k0, res_dg0);
        reinterpret_cast<float2*>(res)[i * 2 + 1] = float2_fma(diff1, k1, res_dg1);
    }
    for (int i = 0; i < K_TILE_ / 8; ++i) {
        store_256B(&res[i * 8], dg_out_base + i * 8);
    }
}

// Output dk = res + res_dkt + dkt_sub + dk_input  (upper half), bf16 cast.
template <int K_TILE_, int K_OFFSET = 0, typename DK_TENSOR, typename DKT_TENSOR>
__forceinline__ __device__ void
epilogue_output_dk(
    DK_TENSOR& sDK, DKT_TENSOR& sDKT_0,
    int idx_in_warpgroup, int sub_seq_len,
    float* res, float* res_dkt, float* dk_out_base) {
    int row = idx_in_warpgroup % 64;
    if (row < sub_seq_len) {
        for (int i = 0; i < K_TILE_ / 16; ++i) {
            float dk_res_fp32[16];
            for (int j = 0; j < 4; ++j) {
                int base = i * 16 + j * 4;
                float4 dkt_sub  = *reinterpret_cast<float4*>(&sDKT_0(row, K_OFFSET + base));
                float4 dk_input = *reinterpret_cast<float4*>(&sDK  (row, K_OFFSET + base));
                for (int k = 0; k < 2; ++k) {
                    float2 sum = float2_add(reinterpret_cast<float2*>(res_dkt + base)[k],
                                            reinterpret_cast<float2*>(&dkt_sub)[k]);
                    sum = float2_add(sum, reinterpret_cast<float2*>(&dk_input)[k]);
                    reinterpret_cast<float2*>(res + base)[k] = float2_add(
                        reinterpret_cast<float2*>(res + base)[k], sum);
                    reinterpret_cast<float2*>(&dk_res_fp32[j * 4])[k] =
                        reinterpret_cast<float2*>(res + base)[k];
                }
            }
            store_256B(&dk_res_fp32[0], dk_out_base + i * 16);
            store_256B(&dk_res_fp32[8], dk_out_base + i * 16 + 8);
        }
    }
}

}  // namespace sm90_bwd
