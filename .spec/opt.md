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
  
之前的结论：
两轮优化后报告：
当前状态: commit 6ae1cb7 (已push), 从 0.20x -> 0.22x FLA (+10%)
未达成 1.5x 目标。 两个优化 agent 的共同结论:
微优化已经穷尽：寄存器压力、launch_bounds、setmaxnreg 都只能拿到 +10%
WGMMA 机械替换不可行：当前算法每个 KG sub-tile (s=0..5) 只贡献 dQ 的一个 16-row 子块 (每行 gn 缩放烤进 B operand) ，WGMMA 64-row M 维度会浪费 75% lane, 导致 ~10x 多余 FLOP
真正 1.5x 路径 需要 ~4-5 天算法重构:
Prep 端把 B operand 重新组织为 64xK K-major SW128 swizzled tiles (不再按 row-block 拆分)
Prep 端预转置 dAqk/dAkk 到 swizzled smem
MMA 改写为 warpgroup_arrive/commit_batch/wait pipeline
causal mask 在 Prep 端 zero A 行
取消 dK_lower 独立 MMA, Prep 端用 gate scale 解析重算
可能还要合并 CE+MMA WG 减少 sync overhead
ncu 证据 (T=1024): tensor pipe 0.61%, long-scoreboard stall 66.4%, 784B/thread 栈溢出, smem 209KB/block 限制 occupancy 到 19.9%.
