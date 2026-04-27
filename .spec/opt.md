我需要对 kda_bwd_intra_sm90.cu 里面的 kernel 进行 refactor，最终目标是为了优化性能，我希望能调到 FLA 的 1.5x，目前是 1.3x。
一定要先 refactor 然后再保证性能不变的情况一直优化到 1.5x。

refator 任务：
（1）增加 ccache 加速编译
（2）SM80.mma 的执行，必须参考 csrc/sm90/collective/mainloop_kda_fwd.hpp:1437-1944 行的实现；
（3）KG，QG，KBG 的计算，必须参考 kda-backward-internal-feat-bwd_intra_opt/csrc/kda_bwd/kda_bwd_intra_sm100.cu
WarpRole::ComputeEpilogue 里面的计算方式，很多计算算子定义在 kda-backward-internal-feat-bwd_intra_opt/csrc/kda_bwd/helpers.h 里面；最终写入 gmem 你也需要参考 WarpRole::ComputeEpilogue。

你需要确保你可以通过单元测试：
python tests/test_kda_bwd_intra.py
且保证benchmark性能不会下降：
python benchmarks/bench_kda_bwd_intra.py

接下来我需要你继续进行性能优化，请使用 ncu 指令自行进行调试，我需要你调到 FLA 的 1.5x 的性能以上才能停止任务：

当你写完了一个你觉得已经完备的版本后，进行

（1）source .venv/bin/activate && pip install -e . --no-build-isolation 进行安装
（2）使用 ncu 获取报告分析；
（3）修改代码，并运行单测和benchmark；
（4）当结果 <1.5x ncu profile，然后根据结果修改代码，不断循环。 


每当代码有优化时候你就可以只就 git commit 并 push origin main，当你发现优化无效就回退到上一个有效的版本。
CUDA_VISIBLE_DEVICES 设置在 7上执行，防止冲突
wgmma 用不了，只能使用同步的 sm80 mma
ncu gui report 顺便也生成到当前目录下，方便我查看

过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。 过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。
如果我有insight ，会在这个文档更新，你定期看看。

ncu 报告请参考生成：
(1)生成带时间戳的GUI报告，方便我查看
  source .venv/bin/activate && CUDA_VISIBLE_DEVICES=7 sudo -E env "PATH=$PATH" $(which ncu) --kernel-name "regex:kda_bwd_intra_kernel_sm90" --launch-skip 3  \
  --launch-count 1 --set full -o bwdddd-$(date) python scripts/ncu_bwd_intra_sm90.py --B 4 --T 2048 --H 4 --D 128 
（2）通过一下命令获取文字结果，进行分析
  ncu -i bwdddd.ncu-rep --section SpeedOfLight --csv 
  ncu -i bwdddd.ncu-rep --section MemoryWorkloadAnalysis --csv
  ncu -i bwdddd.ncu-rep --section Occupancy --csv   
  ncu -i bwdddd.ncu-rep --section SchedulerStats --csv
  
没有完成性能优化（1.5x）前，不要咨询我，我有insight 会在这个文档更新，你就不断尝试就行。你只要保证可以随时可以回退就行，你可以随意发挥。
没有完成性能优化（1.5x）前，不要咨询我，我有insight 会在这个文档更新，你就不断尝试就行。你只要保证可以随时可以回退就行，你可以随意发挥。
没有完成性能优化（1.5x）前，不要咨询我，我有insight 会在这个文档更新，你就不断尝试就行。你只要保证可以随时可以回退就行，你可以随意发挥。

---

## NCU 对比分析 insight (2026-04-27)

报告: `fla-bwd-20260427-195452.ncu-rep` vs `cula-bwd-20260427-195517.ncu-rep` (B=4 T=2048 H=4 D=128, sm_90)。
注意 FLA 用 autotune 12 配置，NCU `--launch-skip 3` 抓到的 294us 是 autotune 扫描期非最优 config，不是稳态时长，**仅看结构性指标**。

### 关键差距
| 指标 | FLA | cuLA | 含义 |
|---|---|---|---|
| Bank 冲突 wavefronts | 2.31M (9%) | **5.37M (35%)** | cuLA smem 访问非常冲突 |
| Load conflicts | 低 | **33%** Est.+29.6% | |
| Store conflicts | 低 | **46%** Est.+41.6% | epilogue scatter 是元凶 |
| L1 Hit Rate | 51% | **12%** | cuLA 重复 LDS 同一数据 |
| 主导 stall | LongScoreboard 2.5 | **Short SB 4.1 + MIO 1.5** | smem port 排队 |
| Warp Cycles/Issued | 7.45 | **18.83** | cuLA 每条指令等更久 |
| LSU 利用率 | 21% | **44%** | LSU 拉满 |
| Tensor (FP) | 1.15% | 3.9% | cuLA mma 用得多 |
| Reg/线程 | 256 | 56 | cuLA occupancy 75% vs 25% |
| 总指令 | 106M | 60.6M | cuLA mma 替代 FFMA，少 1.75x |

### 根因
1. **STS scatter** (`sStage(row, col) = ...`) C-layout 4 路寄存器写到非对齐 row/col → store 冲突 46%。
2. **B operand 用标量 LDS** (`SmemLayoutB_op` stride-17, `S2RAtomB = UniversalCopy<float>`)：每 mma k-loop 8 次标量加载，banks 不分散。
3. cuLA 4 warp 各自 LDS 同一 KG/QG → L1 命中率仅 12%。FLA 用 LDSM 一次多寄存器加载。

### 优化优先级（按 NCU Est.Speedup）
1. **#1 Epilogue R2S 用 STSM/swizzled C-layout** 替换 `sStage(row,col)=` scatter — Est +41.6%
2. **#2 B operand 改用 LDSM + swizzled bf16 layout** (参考 mainloop_kda_fwd.hpp:1437-1944, helpers.h `convert_bf16_to_tf32_operandB_layout`) — Est +29.6%
3. **#3 Phase 2 transposed dA loads 用 `ldmatrix.x4.trans`** 替代标量 transpose-on-write
4. **#4 A operand 也走 LDSM**, dA 写入 smem 时即转 bf16 → smem 容量减半
5. **#5 FFMA fusion** — Est +6.7%
6. **#6 Uncoalesced gmem epilogue** — Est +5.9%

### 总策略
不要散点优化 (上限 +20%)。**核心动作：把 SM80 mma operand A/B/C 全部切到 LDSM/STSM + swizzled smem**，正是 spec 指向 `mainloop_kda_fwd.hpp:1437-1944` 的原因。预期：bank 冲突 → <10%，LSU → 25-30%，Short SB stall 减半，Warp Cycles/Issued 18.8 → 10。综合预期跨过 1.5x。
