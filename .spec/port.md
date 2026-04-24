我需要把SM100里面 WarpRole::ComputeEpilogue 的逻辑 port 到 sm90 的 prep 里面，barrier 目前 sm90 比较粗粒度，优先以 sm90 的 barrier 为主，sm100 mask 不需要 port，其他功能我目测都是需要复用。
当你写完了一个你觉得已经完备的版本后（你还需要把比较重要的结果都打印出来方便 debug，不过打印太大），进行

（1）source .venv/bin/activate && pip install -e . --no-build-isolation 进行安装
（2）根据 debug 打印的结果进行调试，tests/test_kda_bwd_intra_sm90.py 有参考的正确代码

MMA 不需要实现，我会在后面继续实现。
过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。 过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。
上一个 commit 的代码不能保证可以运行成功，你直接调试就可以，然后 MMA 需要发射的 barrier 你可以mock得加上去，用于通过这个测试，然后还有如果 smem 不够的话，就先不考虑这个了，感觉 sm90 需要调整结构了，或许只能直接在里面算了。
```
array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dq;    // dQ (intra) + DKT_0
array_aligned<float, cosize_v<SmemLayoutInputFP32>> smem_mma_dq2;   // dQ2 (inter) + DKT_1
```