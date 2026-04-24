# SM90 Prep vs SM100 ComputeEpilogue porting 对照

## 架构差异

| 维度 | SM90 (当前) | SM100 (代码库) |
|------|------------|----------------|
| Thread alloc | 256-thread Prep + 128-thread MMA WG | 2×128 ComputeEpilogue + 2×128 MMA WG |
| K 拆分 | 不拆分，256 thread stride 覆盖全部 32 | WG0/WG1 各做 K_TILE/2=16 (HALF_K) |
| dA mask/transpose | MMA 内联判断因果 mask | ComputeEpilogue 用 `mask_A_tensor/mask_At_tensor` 预处理到 TMEM |
| TMEM | 无（直接去 smem） | 有（MMA C 算 TMEM，epilogue 读 TMEM 做 scale/combine） |
| Beta 加载 | Load warp 内联 `beta_smem[buf_idx_A]` | Empty warp 独立加载 + `bar_dA_mask_ready` 信号 |
| dB reduce | Tile loop 尾做 `db_accum + db_inter` | 每 k-iteration 内 `WG0 partial → smem → WG1 accum` |
| dK 组装 | MMA 直出 `dk_lower + dk_upper` | MMA 出 `dq`/`dkt` 到 TMEM，CE 读 TMEM+exchange+scale+输出 |

---

## SM100 ComputeEpilogue 的完整 barrier pipeline

```
ComputeEpilogue (per tile):

Step 1:  wait bar_load_dA_ready[A_phase]
         ── 等 Load warp：tile_id + dA TMA done ──

Step 2:  check sentinel (tid >= total_tiles ? break)

Step 3:  mask_A_tensor ( software 因果 mask dA → TMEM )

Step 4:  arrive bar_dA_ready[buf_idx_A]
         ── 通知 MMA：dA 已就位 ──

Step 5:  wait bar_dA_mask_ready[0], tile_phase
         ── **实际语义：等 Empty warp 的 beta_smem 就绪**
         （本应是 bar_beta_ready，名字是历史遗留）

Step 6:  per-ki loop:
          6a. wait bar_load_kg_ready[buf], local_phase
              ── 等 Load warp Q/K/G TMA ──

          6b. Construct KG_intra/inter + QKG_intra/inter
              （和 SM90 Prep elementwise 前半段一致）

          6c. arrive bar_kg_all_ready, bar_qkg_all_ready
              ── 通知 MMA：B operands 就绪 ──

          6d. [first k_idx only] mask_At_tensor (transpose mask → TMEM)
              arrive bar_dAt_ready[buf_idx_A]

          6e. epilogue_compute_intra_scale (读 sG 算 scale)
              epilogue_apply_dq_intra     (把 MMA TMEM dq 读回 register × scale)
              epilogue_combine_dq_inter   (再 × inter scale)
              **（此时 res[] 内等价于 SM90 的 dq_intra）**

          6f. epilogue_accumulate_db (dk_lower * beta × K, 攒 db)
              epilogue_output_dq       (dq → bf16 HBM)

          6g. (WG0 partial db) → named barrier exchange → (WG1 accum db)

          6h. epilogue_compute_dkt_scale, wait bar_dkt_done
              epilogue_process_dkt (读 TMEM dkt × scale)

          6i. named barrier exchange_res/res_dkt
              epilogue_output_dg, epilogue_output_dk
              **（dK 来自 dkt exchange + sDK input）**

          6j. arrive bar_dvalue_free[buf_idx_value]
              b_phase ^= 1; state_phase ^= 1 << buf_idx_value;

Step 7:  output db (WG1 final)

Step 8:  state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE)
         buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A
```

---

## SM90 Prep 现有实现（无需改动即可覆盖的）

```
Prep warp (256 threads):

Step 1:  wait bar_dA_ready, phase_dA
         ── 等价 SM100 Step 1+4（dA loaded）──

Step 2:  zero db_accum
         zero G padding rows (if sub_seq_len < T_TILE)

Step 3:  per-ki loop:
          3a. wait bar_qkg_ready[cur_buf], phase_qkg
              ── 等价 6a（等 Load warp TMA Q/K/G/dQ/dK/dG）──

          3b. Construct KG / QG / KBG (elementwise)
              ── 等价 6b ──

          3c. arrive bar_kg_ready[], bar_qg_ready[], bar_kbg_ready[]
              ── 等价 6c ──

          3d. wait bar_mma_ki_ready[cur_buf], phase_mma_ki
              ── 等价 6e+6f+6h+6i 前的等待 ──

          3e. Epilogue compute per element:
              dq_out = dq_inter + dq_intra
              dk_lower_beta = dk_lower * beta
              dk_out = dk_inter + dk_lower_beta + dk_upper
              dg_out = dg_inter + q_val*dq_intra + (dk_lower_beta - dk_upper)*k_val
              atomicAdd(db_accum[row], dk_lower * k_val)
              ── **一次 kernel pass 完成 SM100 的 dq/dk/dg/db 全部输出** ──

          3f. arrive bar_buf_free[cur_buf]
              phase_qkg[cur_buf] ^= 1
              phase_mma_ki[cur_buf] ^= 1

Step 4:  db_out = db_accum + db_inter (if prep_tid < sub_seq_len)
         phase_dA ^= 1
         arrive bar_epilogue_done
```

---

## 需要 port 的 barrier 语义

| SM100 barrier | SM90 是否需要新增 | 说明 |
|---------------|-------------------|------|
| `bar_load_dA_ready[A_phase]` | **已有** | Load warp 写完 tile_id + dA arrive 的 |
| `bar_dA_ready[buf_idx_A]` | **已有** | MMA 等 dA 就绪 |
| `bar_dA_mask_ready[0]` | **不需要** | 等 beta；SM90 Load warp 直接内联加载 beta，无 Empty warp |
| `bar_kg_all_ready / bar_qkg_all_ready` | **不需要** | MMA 等 B operands；SM90 拆成 3 个独立 `bar_kg/qg/kbg_ready` |
| `bar_dAt_ready[buf_idx_A]` | **不需要** | 等 transpose dA；SM90 MMA 内联 causal mask，无 TMEM transpose 步骤 |
| `bar_dq_done / bar_dkt_done` | **已有** | SM90 对应 `bar_mma_ki_ready[cur_buf]` |
| `bar_dvalue_free[buf_idx_value]` | **已有** | SM90 对应 `bar_buf_free[cur_buf]` |

---

## porting 结论

**不用 port。SM90 Prep 现有的 epilogue 已经是 ComputeEpilogue 的功能子集（且更简洁）。**

- `ComputeEpilogue` 里的 **dA mask/transpose**（`mask_A_tensor`、`mask_At_tensor`）明确要求跳过。
- **Beta 加载**：SM90 由 Load warp 直接填 `beta_smem`，不需要 Empty warp 或 `bar_dA_mask_ready` 信号。
- **dB reduce**：SM90 是在 tile loop 末尾做一次 `db_accum + db_inter`，不需要 intra-tile WG0/WG1 exchange。
- **dK/dG 输出**：SM90 Prep 在 `wait bar_mma_ki_ready` 后直接读 `smem_dq_out/dk_out/dk_upper_out` 写 HBM，没有 TMEM scale/exchange 中间态。

唯一建议：确认 SM90 `bar_dA_ready`（Prep 消费 dA 的信号）和 SM100 `bar_dA_ready`（MMA 消费 dA 的信号）的 arrive count 语义是否一致。SM90 当前 `init_barrier(bar_dA_ready, ...)` 没有看到（可能缺少），需要检查是否漏了 `arrive` 端。

---

## SM100 elementwise 构造细节（`setup_qkg_intra` 示例）

以下代码位于 SM100 `compute_epilogue_body` 中，功能上**完全等价**于 SM90 Prep 的 KG/QG/KBG elementwise 构造。

```cpp
// === COMPUTE: qkg_intra (non-overlapping rows only) ===
{
    float2 beta[4];
    if constexpr (WG_IDX == 0) {
        float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
        int x3 = idx_in_warpgroup / 8 + 48;
        if (x3 < sub_seq_len) beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));
        setup_qkg_intra<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(
            sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta[3], gn1, 2);
    } else {
        ...
    }
}
```

### `setup_qkg_intra` 对应的数学公式

```cpp
// 输入：行 x = idx_in_warpgroup / 8 + tile_j * 16
//       列 y = idx_in_warpgroup % 8 * 4 （连续4列）
//       gn  = G_norm for this sub-tile（即 exp2f 中的参考 G 值）
//       beta = beta[x]
//
// 逐(col)计算：
//   diff = G[x, y:y+4] - G_norm[y:y+4]
//   scale = exp2f(diff) = exp2f(G[x, :] - G_norm[:])
//
// 输出 QG:    QG[x, :] = Q[x, :] * scale
//             → 写入 sQKG_intra(y, idx_in_warpgroup / 8)
//
// 输出 KBG:   KBG[x, :] = K[x, :] * scale * beta[x]
//             → 写入 sQKG_intra(y, idx_in_warpgroup / 8 + 16)
//
// 若 x >= sub_seq_len（padding 行），写全零
```

等价于 SM90 Prep 中的这两段：

```cpp
// QG[i, d] = Q[i, d] * exp2f(G[i, d] - G_norm[d])
float q_val = static_cast<float>(q_ptr[i * K_TILE + d]);
qg_ptr[i * K_TILE + d] = q_val * safe_exp2f(g_val - g_norm);

// KBG[i, d] = K[i, d] * beta[i] * exp2f(G[i, d] - G_norm[d])
float k_val = static_cast<float>(k_ptr[i * K_TILE + d]);
kbg_ptr[i * K_TILE + d] = k_val * beta_val * safe_exp2f(g_val - g_norm);
```
