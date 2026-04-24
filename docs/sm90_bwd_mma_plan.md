# SM90 KDA Backward MMA Implementation Plan

## Context

SM90 backward kernel (`csrc/kda/sm90/bwd/kda_bwd_intra_sm90.cu`) 已有完整数据流基础设施：
- TMA 加载、B-operand 计算 (KG/QKG)、epilogue 函数均已完成
- **MMA warpgroup 完全为空**（128 线程 / 4 warp），所有 MMA 结果读取自 `smem_mma_zero` 占位零缓冲
- 需要：用 SM80 `mma.sync` (TF32 16x8x8) 填充 MMA warpgroup，计算 dQ / dQ2 / dKt

SM100 参考实现使用 UMMA + TMEM + 硬件 mask，SM90 上不存在。SM90 用 `mma.sync` 替代，causal mask 通过寄存器置零实现。

---

## 1. 核心参考：前向 kernel 的 MMA 模式

前向 `mainloop_kda_fwd.hpp` 中 `qk_kk_subchunk_mma_and_store` 展示了 SM90 上完整的 SM80 mma.sync 使用模式。**后向 kernel 应复用相同的模式。**

### 关键类型与配置

```cpp
using MMA = SM80_16x8x8_F32TF32TF32F32_TN;
using TiledMma_SubChunk =
    decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _2, _1>>{}, TileShape_SubChunk{}));
```

- MMA atom: `SM80_16x8x8_F32TF32TF32F32_TN` → M=16, N=8, K=8 每次 mma.sync
- TiledMma 通过 `Layout<Shape<_1, _2, _1>>` 沿 N 复制 2 份，2 warp 协作
- 前向用 2 warp 处理 16×16 子块；**后向用 4 warp 沿 M 分布**，覆盖 M=64 行

### 前向 S2R (Shared-to-Register) 模式

前向使用 **BF16 MMA layout 做 LDSM 加载**，然后 warp shuffle 转 TF32 layout：

```cpp
// BF16 MMA (same shape 16x8x8) used only for creating LDSM-compatible tiled copies
using MMA_BF16 = SM80_16x8x8_F32BF16BF16F32_TN;
using TiledMma_BF16_SubChunk =
    decltype(make_tiled_mma(MMA_BF16{}, Layout<Shape<_1, _2, _1>>{}, TileShape_SubChunk{}));

// S2R: LDSM copies using BF16 MMA layout for efficient ldmatrix loads
using CopyQKAtom_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
auto Q_tiled_copy = make_tiled_copy_A(CopyQKAtom_LDSM{}, tiledmma_bf16_subchunk);
auto Kt_tiled_copy = make_tiled_copy_B(CopyQKAtom_LDSM{}, tiledmma_bf16_subchunk);

// R2S: STSM for accumulator store
using CopyOp_R2S = SM90_U32x2_STSM_N;
auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOp_R2S, Element>{}, tiledmma_subchunk);
```

**后向 kernel 的差异**：
- 后向的 A operand 是 dA (fp32, 在 SMEM 中用 `SmemLayoutDA`)
- 后向的 B operand 是 KG/QKG (tf32, 在 SMEM 中用 `SmemLayoutMatBTF32Tranposed`)
- 后向不需要 BF16→TF32 的 layout 转换（A 是 fp32→tf32 截断，B 已经是 tf32）
- 因此简化：A operand 用 `AutoVectorizingCopy` 加载 + 手动 tf32 截断，B operand 用 `AutoVectorizingCopy` 加载

### 前向 Gemm 循环模式

```cpp
// allocate acc [16, 16]
Tensor tQKrQK_r_c = partition_fragment_C(tiledmma_subchunk, select<0,1>(TileShape_SubChunk{}));
clear(tQKrQK_r_c);

// for loop head dim (K 维度迭代)
CUTE_NO_UNROLL
for (int j = 0; j < NK; ++j) {
    // S2R operand A (行 r, K 切片 j)
    auto [tQKrQ_r_j, tQKrK_r_j, tArAfirst_r_j_kt] = s2r_compute_subchunk_operandA(r_, j, j0, j1);
    // S2R operand B (列 c, K 切片 j)
    auto tQKrKt_c_j = s2r_compute_subchunk_operandB(c_, j, j0, j1, tArAfirst_r_j_kt);

    // gemm accumulate
    gemm(tiledmma_subchunk, tQKrQ_r_j, tQKrKt_c_j, tQKrQK_r_c);
}

// epilogue + R2S
r2s_subchunk_acc(r_, c_, tQKrQK_r_c, tKKrKK_r_c);
```

### 前向 R2S (Register-to-Shared) 模式

```cpp
auto r2s_subchunk_acc = [&](auto r_, auto c_, auto const& tQKrQK, auto const& tKKrKK) INLINE_LAMBDA {
    // R2S QK: fp32 acc → bf16 → STSM
    Tensor sQK_r_c = sQK_slice(_, _, r_, c_);
    Tensor tQKsQK_r_c = O_thr_copy.partition_D(sQK_r_c);
    Tensor tQKrQK_cv = O_thr_copy.retile_S(tQKrQK);
    auto tQKrQK_cvt_cv = make_fragment_like<Element>(tQKrQK_cv);
    cute::transform(tQKrQK_cv, tQKrQK_cvt_cv, [](auto v) { return Element(v); });
    copy(O_tiled_copy, tQKrQK_cvt_cv, tQKsQK_r_c);
};
```

**后向差异**：后向 MMA 结果 (dQ/dQ2/dKt) 是 fp32，要写回 fp32 SMEM 缓冲，不需要 bf16 转换。直接用 `AutoVectorizingCopy` 或带 swizzle 的 copy 写回。

---

## 2. TiledMMA 定义

后向 kernel 用 4 warp 沿 M 维分布（每 warp 覆盖 M=16 行，总共 M=64），N 维不需要复制（N=K_TILE=32 通过外部循环覆盖）：

```cpp
// 4 warp 沿 M 分布
using TiledMMA_BWD = decltype(make_tiled_mma(
    SM80_16x8x8_F32TF32TF32F32_TN{},
    Layout<Shape<_4, _1, _1>>{}  // 4 warp 沿 M
));
// size(TiledMMA_BWD{}) == 128 = 4 warp × 32 threads/warp ✓
```

在 MMA warpgroup 体内：
```cpp
auto tiled_mma = TiledMMA_BWD{};
const int thread_idx_in_wg = thread_idx - MMA_THREAD_OFFSET;  // 0..127
auto thr_mma = tiled_mma.get_thread_slice(thread_idx_in_wg);
```

MMA 产生的 C fragment 形状：
- 每个 warp: `partition_fragment_C` → Shape <_16, _8> per tf32 atom, 外部循环覆盖
- 4 warp 共同覆盖 M=64, N=8 per K_STEP
- N=32 通过 4 次外部 N 循环覆盖（或直接 partition 更大的 N）

---

## 3. SharedStorage 改动

替换 `smem_mma_zero` 为 3 个实际 MMA 结果缓冲：

```cpp
// 删除:
// array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_zero;  // 8 KB

// 新增:
array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dq;    // dQ (intra), ~8 KB
array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dq2;   // dQ2 (inter), ~8 KB
array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dkt;   // dKt, ~8 KB
```

dkt 交换缓冲复用 `smem_mma_dq` 和 `smem_mma_dq2`（dQ/dQ2 在 epilogue 消费后不再需要）。

**SMEM 预算**: 净增 3×8KB - 8KB = 16KB。总计约 217KB < 228KB 限制。

**删除零初始化代码**（kernel 入口处第 585-595 行的 `smem_mma_zero` memset 块）。

---

## 4. Barrier 改动

```
// 删除:
// alignas(16) cute::uint64_t bar_mma_ki_ready[NUM_BUF];  // dQ+dK per-ki output ready

// 新增 (phase-tracked, single-buffered):
alignas(16) cute::uint64_t bar_mma_dq_done;   // MMA(128) → Prep: dQ+dQ2 ready
alignas(16) cute::uint64_t bar_mma_dkt_done;   // MMA(128) → Prep: dKt ready
```

初始化：
```cpp
cute::initialize_barrier(smem->bar_mma_dq_done, 128);
cute::initialize_barrier(smem->bar_mma_dkt_done, 128);
```

Prep 端读取前需 wait：
- 读取 dQ/dQ2 前：`wait_barrier(smem->bar_mma_dq_done, b_phase)`
- 读取 dKt 前：`wait_barrier(smem->bar_mma_dkt_done, b_phase)`

---

## 5. A Operand 加载 + Causal Mask

### KG Phase: dAqk 做矩阵 A

dAqk 在 SMEM (`smem_daqk[buf_idx_A]`) 中，layout 为 `SmemLayoutDA` (64×64 fp32)。

加载方式：用 `thr_mma.partition_A(sDA)` + `copy()` 或手动从 SMEM 读入 A fragment。

**Causal mask 规则**：
- A[i][j] = dAqk[i][j]，**当 row < col 时置零**（causal 上三角屏蔽）
- 当 row >= sub_seq_len 或 col >= sub_seq_len 时置零

实现：在 partition_C 的坐标映射 (identity tensor) 中获取 (row, col)，对不满足条件的 fragment 元素置零。

```cpp
auto cM = make_identity_tensor(make_shape(Int<T_TILE>{}, Int<K_SUB>{}));  //沟配当前子块的 K 范围
auto tAcM = thr_mma.partition_A(cM);  // A operand 的坐标映射
// 加载后遍历:
for (int i = 0; i < size(tArA); ++i) {
    auto [row, col] = tAcM(i);  // 全局坐标
    if (row < col || row >= sub_seq_len || col >= sub_seq_len) {
        tArA(i) = 0.0f;
    }
}
```

### QKG Phase: dAqk^T + dAkk^T 做矩阵 A

转置读取，从同一 SMEM 缓冲以转置索引访问。

- QKG tile 的 K 维前 SUB_T_TILE 行对应 dAqk^T（乘 QG），后 SUB_T_TILE 行对应 dAkk^T（乘 KBG）
- Causal mask: 当 **col_orig > row_orig**（下三角不含对角线）时置零
- 即：转置矩阵中的 (m, k) → 原矩阵 (k, m)，mask 条件为 k > m

---

## 6. B Operand 加载

B operand（KG / QKG tiles）已由 Prep warpgroups 写入 SMEM。

```cpp
auto sB = make_tensor(make_smem_ptr(smem->kg_all.intra[idx].data()), SmemLayoutMatBTF32Tranposed<N>{});
auto tBsB = thr_mma.partition_B(sB);
auto tBrB = make_fragment_like(tBsB);
cute::copy(tiled_copy_B, tBsB, tBrB);  // SMEM → 寄存器
```

`SmemLayoutMatBTF32Tranposed` 使用 UMMA swizzle layout，CuTe 内部处理 swizzle 解码。

---

## 7. 累加器写回 SMEM

参照前向 `r2s_subchunk_acc` 模式，使用 CuTe `copy` 原语：

```cpp
// 写回 dQ 结果
auto sDQ = make_tensor(make_smem_ptr(smem->smem_mma_dq.data()), SmemLayoutInputFP32{});
auto tCsDQ = thr_mma.partition_C(sDQ);
cute::copy(tCrDQ, tCsDQ);  // 寄存器 → SMEM
fence_view_async_shared();
```

CuTe 自动处理 fragment→SMEM 的坐标映射，无需手动 `sg_acc_mapping.md` 中的公式。

---

## 8. MMA Warpgroup 完整框架

```cpp
} else if (role == WGRole::MMA) {
    cutlass::arch::warpgroup_reg_alloc<REG_MMA>();

    const int thread_idx_in_wg = thread_idx - MMA_THREAD_OFFSET;  // 0..127
    auto tiled_mma = TiledMMA_BWD{};
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx_in_wg);

    for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
        // wait dA ready
        int A_phase = (state_phase >> (buf_idx_A + NUM_BUF)) & 1;
        cute::wait_barrier(smem->bar_load_dA_ready[buf_idx_A], A_phase ^ 1);

        // decode tile coordinates
        int tid = tile_scheduler.get_current_tile_id();
        auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
        int sub_seq_len = ...;

        int b_phase = 0;

        for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {

            // ══════════ KG PHASE → dQ + dQ2 ══════════
            cute::wait_barrier(smem->bar_kg_all_ready, b_phase);

            // --- Intra: dQ = dAqk × KG (3 sub-tiles: tile_j = 0, 1, 3) ---
            auto tCrDQ = partition_fragment_C(tiled_mma, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
            clear(tCrDQ);

            for (int tile_j : {0, 1, 3}) {
                // load A from smem_daqk[:, KG_col_range] + apply causal mask
                // load B from smem->kg_all.intra[tile_j]
                // gemm(tiled_mma, tArA, tBrB, tCrDQ)
            }

            // R2S: write tCrDQ → smem_mma_dq
            fence_view_async_shared();

            // --- Inter: dQ2 = dAqk × KG (4 sub-tiles: tile_i = 0..3) ---
            auto tCrDQ2 = partition_fragment_C(tiled_mma, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
            clear(tCrDQ2);

            for (int tile_i = 0; tile_i < 4; ++tile_i) {
                // load A from smem_daqk[:, KG_col_range] + apply causal mask
                // load B from smem->kg_all.inter[tile_i]
                // gemm(tiled_mma, tArA, tBrB, tCrDQ2)
            }

            // R2S: write tCrDQ2 → smem_mma_dq2
            fence_view_async_shared();
            cute::arrive_barrier(smem->bar_mma_dq_done);  // signal Prep

            // ══════════ QKG PHASE → dKt ══════════
            cute::wait_barrier(smem->bar_qkg_all_ready, b_phase);

            auto tCrDKT = partition_fragment_C(tiled_mma, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
            clear(tCrDKT);

            // Intra: 3 sub-tiles with transposed A (dAqk^T + dAkk^T)
            for (int tile_j : {0, 1, 3}) {
                // load transposed A from smem_daqk/dakk + apply transposed causal mask
                // load B from smem->qkg_all.intra[tile_j]
                // gemm(tiled_mma, tArA, tBrB, tCrDKT)
            }

            // Inter: 4 sub-tiles with transposed A
            for (int tile_i = 0; tile_i < 4; ++tile_i) {
                // load transposed A + apply mask
                // load B from smem->qkg_all.inter[tile_i]
                // gemm(tiled_mma, tArA, tBrB, tCrDKT)
            }

            // R2S: write tCrDKT → smem_mma_dkt
            fence_view_async_shared();
            cute::arrive_barrier(smem->bar_mma_dkt_done);  // signal Prep

            b_phase ^= 1;
        }
        // advance buf tracking same as Load/Prep
    }
}
```

---

## 9. Epilogue Call Site 更新

`prep_compute_epilogue_body` 中所有 `smem_mma_zero` 指向改为实际缓冲：

| 当前代码 | 改为 |
|---------|------|
| `make_smem_ptr(smem->smem_mma_zero.data())` 用于读 dQ | `make_smem_ptr(smem->smem_mma_dq.data())` |
| `make_smem_ptr(smem->smem_mma_zero.data())` 用于读 dQ2 | `make_smem_ptr(smem->smem_mma_dq2.data())` |
| `make_smem_ptr(smem->smem_mma_zero.data())` 用于读 dKt | `make_smem_ptr(smem->smem_mma_dkt.data())` |
| `make_smem_ptr(smem->smem_mma_zero.data())` 用于 sDKT_0 | `make_smem_ptr(smem->smem_mma_dq.data())` (复用) |
| `make_smem_ptr(smem->smem_mma_zero.data())` 用于 sDKT_1 | `make_smem_ptr(smem->smem_mma_dq2.data())` (复用) |

新增 wait_barrier：
- 读 dQ/dQ2 前：`cute::wait_barrier(smem->bar_mma_dq_done, b_phase);`
- 读 dKt 前：`cute::wait_barrier(smem->bar_mma_dkt_done, b_phase);`

Epilogue 函数本身无需修改（接受 tensor 引用，layout-agnostic）。

---

## 10. 实施顺序

| # | 任务 | 文件 |
|---|------|------|
| 1 | SharedStorage: 替换 smem_mma_zero → smem_mma_dq/dq2/dkt | kda_bwd_intra_sm90.cu |
| 2 | Barrier: bar_mma_ki_ready → bar_mma_dq_done + bar_mma_dkt_done | kda_bwd_intra_sm90.cu |
| 3 | TiledMMA_BWD 类型定义 + partition 辅助 | kda_bwd_intra_sm90.cu |
| 4 | A-loading helper + causal mask (KG phase & QKG phase) | kda_bwd_helpers.h |
| 5 | 累加器 SMEM 写回 (cute::copy) | kda_bwd_intra_sm90.cu |
| 6 | MMA warpgroup 循环体 (KG + QKG phases) | kda_bwd_intra_sm90.cu |
| 7 | Prep epilogue call sites 更新 + wait_barrier | kda_bwd_intra_sm90.cu |
| 8 | 删除零初始化代码 | kda_bwd_intra_sm90.cu |
| 9 | 编译验证 + 正确性测试 | - |

---

## 11. 关键注意事项

1. **K 循环**: 每次 MMA 的 K 维度 = sub-tile 宽度 (16/32/48)，需 ceil(K_sub/8) 次 mma.sync 调用
2. **N 循环**: N=32 拆为 4 个 N=8 子块 (SM80_16x8x8 的 N=8)，或直接用 TiledMma partition 更大的 N
3. **QKG Phase A 交织**: QKG tile K 维前半 = QG (乘 dAqk^T)，后半 = KBG (乘 dAkk^T)
4. **sub_seq_len 处理**: 超出 sub_seq_len 的行/列在 A fragment 中置零
5. **注册器压力**: 保持累加器合理分块，REG_MMA=168
6. **前向模式关键差异**: 前向用 BF16 LDSM + warp shuffle 转 TF32 layout；后向 A 是 fp32 SMEM，B 是 tf32 SMEM，不需要 BF16→TF32 的 layout 转换

## 12. 验证

1. 逐步：先 KG Phase (dQ)，正确后再 QKG Phase (dKt)
2. 对比 SM100 输出 (小序列长度)
3. KDA_BWD_SM90_DEBUG_PRINT 打印 checksum
4. Causal mask 检查：dQ 上三角应为零