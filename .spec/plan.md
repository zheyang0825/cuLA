# SM90 KDA Backward Intra-Chunk 实施计划

基于 design.md v3 和 SM100 参考实现，分 4 个阶段，每个阶段可独立验证。

> 所有新文件放在 `csrc/kda/sm90/bwd/`，重复代码先 copy，最后处理冗余。

---

## 阶段 1：Kernel 框架 + 基础设施

**目标**：搭建可编译运行的 kernel 骨架，实现数据加载和 WG 分发。

### 需要创建的文件

| 文件 | 说明 |
|------|------|
| `csrc/kda/sm90/bwd/kda_bwd_common.h` | 参数结构体 `KDA_bwd_intra_params` + `NaiveTileScheduler`（copy SM100） |
| `csrc/kda/sm90/bwd/kda_bwd_utils.h` | CHECK_CUDA / KDA_ASSERT 宏（copy SM100 utils.h） |
| `csrc/kda/sm90/bwd/kda_bwd_basic.h` | bf16/tf32 类型定义（copy SM100 basic.h，namespace 改 sm90） |
| `csrc/kda/sm90/bwd/kda_bwd_helpers.h` | SM90 兼容的 helpers：float2 ops、launch_tma_copy、store_128b（从 SM100 helpers.h 裁剪，去掉 tcgen05/tmem/UMMA 相关） |
| `csrc/kda/sm90/bwd/kda_bwd_intra_sm90.cuh` | 前向声明 `run_kda_bwd_intra_sm90()` |
| `csrc/kda/sm90/bwd/kda_bwd_intra_sm90.cu` | SM90 backward intra kernel 主体 |
| `csrc/api/kda_bwd_sm90.cu` | PyTorch C++ API 入口 |
| `tests/test_kda_bwd_intra_sm90.py` | 测试脚本 |

### 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `csrc/api/pybind.cu` | 注册 `chunk_kda_bwd_intra_sm90` |
| `setup.py` | 添加编译源文件 `kda_bwd_intra_sm90.cu` + `kda_bwd_sm90.cu` |

### 具体内容

1. **`kda_bwd_common.h`**：
   - `KDA_bwd_intra_params` 结构体（对齐 SM100 `kda_config.h`）
   - `NaiveTileScheduler`（直接 copy SM100 tile_scheduler.h）

2. **SharedStorage 结构体**（design.md 第 7 节，~201KB）：
   ```
   Q, K 输入 (×2 buf)          : 16 KB   bf16 64×32
   G 输入 (×2 buf)             : 16 KB   fp32 64×32
   dQ, dK, dG inter (×2 buf)   : 48 KB   fp32 64×32
   dAqk + dAkk                 : 32 KB   fp32 64×64
   KG (×2 buf)                 : 16 KB   fp32 64×32
   QG (×2 buf)                 : 16 KB   fp32 64×32
   KBG (×2 buf)                : 16 KB   fp32 64×32
   dQ out (×2 buf)             : 16 KB   fp32 64×32
   dK out (×2 buf)             : 16 KB   fp32 64×32
   标量 + barriers             : ~2 KB
   合计                        : ~201 KB (228 KB 限制内)
   ```

3. **Kernel 框架** — 512 线程 = 4 WG（design.md 第 4 节）：
   ```
   WG0 (128线程) : LdSt — TMA 加载 + 写回
   WG1 (128线程) : MMA  — mma.sync 4-pass
   WG2 (128线程) : Prep — 标量计算
   WG3 (128线程) : Prep — 与 WG2 协作
   ```

4. **LdSt 实现**（design.md 第 4 节 LdSt Warp 分工）：
   - Warp0 (LoadQKG): TMA 加载 Q, K, G + dQ, dK, dG inter（per-ki 双缓冲）
   - Warp1 (LoadDA): TMA 一次性加载 dAqk, dAkk + beta
   - Warp2: 空闲/备用
   - Warp3 (StoreMisc): TMA 写回 dB

5. **Pipeline 初始化**（design.md 第 6 节，9 个 pipeline）

6. **TMA 描述符创建**：复用 SM100 `run_kda_bwd_intra_sm100()` 的 host 端模式

**验证**：kernel 编译运行，debug buffer 写出 dAqk/dAkk 精确匹配 Python 输入。

---

## 阶段 2：Prep 构造 B 操作数 (KG / QG / KBG)

**目标**：Prep WG 正确构造三种 B 操作数，per-subchunk G_norm。

### 具体内容

1. **Prep 等待 dA TMA 就绪**，通知 MMA（`mask_pipeline`）
2. **per-ki 循环中构造 KG**（4 个 sub-tile，各用独立 G_norm）：
   ```
   G_norm_js = G[js * 16, d]    // sub-tile 首行
   KG[j, d] = K[j, d] * exp2f(G_norm_js[d] - G[j, d])
   ```
3. **构造 QG 和 KBG**（gate 方向相反）：
   ```
   G_norm_is = G[is * 16, d]
   QG[i, d] = Q[i, d] * exp2f(G[i, d] - G_norm_is[d])
   KBG[i, d] = K[i, d] * beta[i] * exp2f(G[i, d] - G_norm_is[d])
   ```
4. **双缓冲**：KG/QG/KBG 各 2 个 buffer

### 关键差异 vs SM100

- SM100 KG/QG/KBG 使用转置 layout (`SmemLayoutMatBTF32Tranposed`) 供 UMMA 消费
- SM90 使用普通 row-major fp32 layout，mma.sync 线程从 SMEM 直接加载到寄存器
- SM100 的 `setup_kg_intra` / `setup_qkg_intra` 等函数逻辑可复用，但输出 layout 需适配

**验证**：KG/QG/KBG 写 debug buffer，与 Python reference 比对。

---

## 阶段 3：MMA 4-pass mma.sync 子块循环

**目标**：实现核心 mma.sync 子块循环，产出原始 dQ / dK_lower / dK_upper。

### 具体内容

1. **MMA Atom**：`SM80_16x8x8_F32TF32TF32F32_TN`，128 线程（1 WG）
2. **Pass 1 (dQ)**：
   ```
   for is = 0..3:
       frag_dq = 0
       for js = 0..is:
           load dAqk[is][js] 16×16 from SMEM → reg, 即时 mask (js<is: 全满, js==is: j≤i)
           mma.sync: frag_tmp = dA_sub × KG_js[16×32]
           q_scale = exp2f(min(G[i] - G_norm_js, 0))
           frag_dq += frag_tmp * q_scale
       R2S: frag_dq → smem_dq_out[is*16..is*16+15]
   ```
3. **Pass 2 (dK_lower)**：类似 Pass 1，dAkk，js < is（严格下三角）
4. **Pass 3 (dK_upper_qk)**：
   ```
   for js = 0..3:
       frag_dkt = 0
       for is = js..3:
           load dAqk[is][js] 即时转置 → reg
           mma.sync: frag_tmp = dA_T_sub × QG_is[16×32]
           k_scale = exp2f(clamp(G_norm_is - G[j], -126, 126))
           frag_dkt += frag_tmp * k_scale
   ```
5. **Pass 4 (dK_upper_kk)**：累加到 Pass 3 同一 fragment，dAkk_T × KBG，is > js
6. **Fragment 复用**：Pass 1/2 串行共用 fragment，Pass 3/4 累加
7. **R2S 写出 + `mma_ki_pipeline` 通知 Prep**

### MMA 计数（design.md 3.6 节）

| Pass | 子块 MMA 调用数 |
|------|--------------|
| 1 (dQ) | 1+2+3+4 = 10 |
| 2 (dK_lower) | 0+1+2+3 = 6 |
| 3 (dK_upper_qk) | 4+3+2+1 = 10 |
| 4 (dK_upper_kk) | 3+2+1+0 = 6 |
| **合计/ki** | **32** |
| **4 ki 总计** | **128 次 mma.sync** |

**验证**：MMA 原始输出写 debug buffer，与 Python reference 比对（atol=0.005）。

---

## 阶段 4：Epilogue + 完整输出

**目标**：完成最终梯度计算和输出。

### 具体内容（design.md 2.5 节 + 3.6 节 Epilogue）

1. **Prep 逐 ki epilogue**（`mma_ki_pipeline` 通知后执行）：
   ```
   dB[i] += Σ_d dK_lower[i,d] × K[i,d]      // beta 缩放前
   dK_lower_final = dK_lower × beta[i]
   dK = dK_inter + dK_lower_final + dK_upper
   dQ = dQ_inter + dQ_out                     // gate scale 已在子块循环中完成
   dG = dG_inter + Q × dQ_intra + (dK_intra - dK_upper) × K
   ```
2. **st.global 输出**：Prep 直接写 dQ/dK/dG 到 HBM
3. **dB 输出**：跨 ki 累加后写 SMEM → LdSt/Warp3 TMA store（或直接 st.global）
4. **接入完整 pipeline 同步**

**验证**：与 FLA Triton 参考实现比对：
```python
assert_close("dq", dq_cuda, dq_triton, atol=0.008)
assert_close("dk", dk_cuda, dk_triton, atol=0.008)
assert_close("db", db_cuda, db_triton, atol=0.02)
assert_close("dg", dg_cuda, dg_triton, atol=0.02)
```

---

## 实施注意事项

### SM90 vs SM100 关键差异

| | SM100 | SM90 |
|---|---|---|
| MMA 指令 | UMMA (1 warp, TMEM) | mma.sync (1 WG, 128 线程, 寄存器) |
| 中间结果 | TMEM | SMEM + 寄存器 fragment |
| Causal mask | UMMA 内建 MASK02/MASK13 | 即时 mask（寄存器加载时判断） |
| 转置 | TMEM 内转置 | 寄存器加载时即时转置 |
| B 操作数格式 | tf32 转置 layout | fp32 row-major |
| 标量计算 | CE 256 线程 | Prep 256 线程 |

### 复用策略

- `NaiveTileScheduler`：直接 copy
- `KDA_bwd_intra_params`：直接 copy
- `launch_tma_copy`：直接 copy
- `float2_add/sub/mul/fma`：直接 copy
- `store_128b`：直接 copy
- SM100 `setup_kg_intra` / `setup_qkg_intra` 等函数：逻辑复用，输出 layout 适配
- TMA 描述符创建 host 端代码：直接 copy

### 不复用（SM100 特有）

- `tcgen05_*` / `tmem_*`：SM100 TMEM 专用
- `umma_arrive_noelect`：SM100 UMMA 专用
- `utcmma_ss` / `utcmma_ts`：UMMA wrapper
- `mask_A_tensor` / `mask_At_tensor`：TMEM 预构造 mask（SM90 改为即时 mask）
- `SM100_MMA_TF32_TS_MASK*`：UMMA mask 模式
