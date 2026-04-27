sm90 mma 目前实现的性能很低，请使用 ncu 指令自行进行调试，我需要你调到 FLA 的 1.5x 的性能以上才能停止任务：

当你写完了一个你觉得已经完备的版本后（你还需要把比较重要的结果都打印出来方便 debug，不过打印太大），进行

（1）source .venv/bin/activate && pip install -e . --no-build-isolation 进行安装
（2）根据 debug 打印的结果进行调试，tests/test_kda_bwd_intra_sm90.py 有参考的正确代码
（3）python benchmarks/bench_kda_bwd_chunk_intra.py 
（4）当结果 <1.5x ncu profile，然后根据结果修改代码，不断循环。 


每当代码有优化时候你就可以只就 git commit 并 push origin main，当你发现优化无效就回退到上一个有效的版本。
CUDA_VISIBLE_DEVICES 设置在 7上执行，防止冲突
wgmma 用不了，只能使用同步的 sm80 mma
ncu gui report 顺便也生成到当前目录下，方便我查看

过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。 过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。
如果我有insight ，会在这个文档更新，你定期看看。

  source .venv/bin/activate && CUDA_VISIBLE_DEVICES=7 sudo -E env "PATH=$PATH" $(which ncu) --kernel-name "regex:.*kda_bwd_intra_sm90.*" --launch-skip 3  \
  --launch-count 1 --set full -o bwdddd python scripts/ncu_bwd_intra_sm90.py --B 4 --T 2048 --H 4 --D 128 

  ncu -i bwdddd.ncu-rep --section SpeedOfLight --csv 
  ncu -i bwdddd.ncu-rep --section MemoryWorkloadAnalysis --csv
  ncu -i bwdddd.ncu-rep --section Occupancy --csv   
  ncu -i bwdddd.ncu-rep --section SchedulerStatistics --csv
  
通向 1.5x 的真路径（4-5 天算法重构）
把 B 重组为 64xK_TILE SW128 swizzled tiles (消除 per-row sub-tile 分块)
Prep 端预 zero causal-mask 行
减少 Prep WG 到 384 线程让 setmaxnreg<240> 真正生效
融合 KG+QKG 阶段

---

## Insight (2026-04-27): WGMMA 重构方案 — anchor 拆分

### 问题：为什么之前不能用 WGMMA
KDA 后向 intra-chunk 数学：
```
dQ[i,d] = Σ_j dAqk[i,j] * KG_{anchor_i}[j,d]
        = Σ_j dAqk[i,j] * K[j,d] * exp2(gn_i[d] - G[j,d])
```
- T_TILE=64 chunk 分 4 个 SUB_T_TILE=16 query sub-chunks
- 每个 query sub-chunk i 有自己的 anchor `gn_i[d]`（采样自 sub-chunk 中点 row=i*16+8）
- anchor 把 `exp2(gn_i - G[j,d])` 的指数控制在 ~24 行 gate 累积，避免 overflow/underflow
- 因 anchor 随 i 变 → B operand (KG) 必须 per-(q_block_i, k_row) 预乘 → 6 个 16-row sub-tiles → WGMMA 64-row M 维浪费 75% lane

### 关键恒等式 (online-norm 拆分)
```
exp2(gn_i[d] - G[j,d]) = exp2(gn_REF[d] - G[j,d]) * exp2(gn_i[d] - gn_REF[d])
                          └─ 公共，烤进 B 大 tile ──┘   └─ 输出端 per-row vector rescale ─┘
```
选 chunk 内固定 anchor `gn_REF`（推荐 `gn_2`，chunk 中点）。

### 重构方案

**Prep WG (B operand 预算)**
- 删除 `kg_all` (6 intra + 4 inter sub-tiles, 20KB) 与 `qkg_all` (40KB) 与 `smem_daqk_T` (16KB)
- 新增：
  - `K̃[j,d] = K[j,d] * exp2(gn_REF[d] - G[j,d])`，j∈[0,64) → 单个 64×K SW128 swizzled tile
  - `Q̃[x,d] = Q[x,d] * exp2(G[x,d] - gn_REF[d])`，x∈[0,64) → 一个 64×K tile
  - `K̃β[x,d] = K[x,d] * β[x] * exp2(G[x,d] - gn_REF[d])` → 一个 64×K tile
- 因数值范围最坏 32 行 gate 跨度（chunk 半边），若担心数值精度可改用 **2 anchor**：`gn_1` 服务 i∈{0,1}，`gn_3` 服务 i∈{2,3}，每 anchor 覆盖 ≤16 行（比原 24 行还紧）→ K̃/Q̃/K̃β 各算两份。

**A operand (dAqk/dAkk)**
- Prep 端预 zero causal mask（dAqk[i_row, j_col] = 0 if i_row < j_col 等）→ MMA path branch-free
- 整个 64×64 chunk 一次 WGMMA 喂入，不再 per-block 切

**MMA WG**
- WGMMA m64n128k16 ×（K_TILE/16）累加；M=64 query 全维度，N=128 head dim，K=16 reduce
- 累加器 64×128 留在 register
- Pipeline: `wgmma.fence` / `commit_batch` / `wait_group`

**Epilogue**
- 4 个 per-D 缩放向量 `s_i[d] = exp2(gn_i[d] - gn_REF[d])`, i∈{0,1,2,3}（或 anchor 子集）
- 累加器 lane 按 query sub-chunk i 选对应 `s_i` 做 fmul 后 R2S
- s_i 由 Prep 算好放 smem (~2KB)

**SMEM 预算估算**
- 释放: `kg_all`(20) + `qkg_all`(40) + `smem_daqk_T`(16) ≈ **-76KB**
- 新增: 2× K̃(16) + 2× Q̃(16) + 2× K̃β(16) + s_i(2) ≈ **+50KB**
- 净释放 **~26KB** → 220.9KB → ~195KB → 仍单 block/SM，但若把 fp32 dq_in/dk_in/dg_in 单 buffer 化（-24KB）可挤进双 block

**与 FLA 对齐**
- FLA Triton 在 sm_90 上对 KDA bwd intra 编译到 WGMMA m64n128k16，与本方案 tiling 完全一致
- 数值上：原方案 anchor i 直接烤入 vs 新方案 anchor REF 烤入 + 输出 rescale，结合律下数学等价；fp32 累加误差顺序略不同，rmse 应在同量级 (1e-5)

### 实施 todo
1. 实现 K̃/Q̃/K̃β builder（fused row-loop, 复用现有 `setup_kg/qkg` row 模板）
2. 实现 Prep 端 dAqk/dAkk causal pre-mask
3. 实现 s_i[d] 缩放向量（4 行 × 128 列）
4. 实现 WGMMA m64n128k16 mainloop（参考 csrc/cutlass/include/cute/atom/mma_atom_sm90.hpp）
5. 实现 epilogue R2S 时 per-sub-chunk vector rescale
6. 删除旧的 6+4 sub-tile B operand path、scalar dAkk gather、mask-in-MMA 分支
7. SMEM 重排，确认 ≤ 228KB
8. 数值验证 vs tests/test_kda_bwd_intra_sm90.py
9. NCU profile，目标 ≥1.5x FLA