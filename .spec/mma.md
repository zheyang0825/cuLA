我需要把SM100里面 WarpRole::MMA 的逻辑 port 到 sm90 的 prep 里面，barrier 目前 sm90 比较粗粒度，优先以 sm90 的 barrier 为主。MMA需要使用 sm80.mma, 具体实现逻辑参考 csrc/kda/collective/mainloop_kda_fwd.hpp:1437-1944, 核心是注意 mask 操作在寄存器中操作，不要再 smem 里操作。

当你写完了一个你觉得已经完备的版本后（你还需要把比较重要的结果都打印出来方便 debug，不过打印太大），进行

（1）source .venv/bin/activate && pip install -e . --no-build-isolation 进行安装
（2）根据 debug 打印的结果进行调试，tests/test_kda_bwd_intra_sm90.py 有参考的正确代码

当代码完备的时候你就可以只就 git commit 并 push origin main。
接着进行 benchmark 的测试，如果性能或者精度不如FLA的时候，就自行 ncu 拉去报告，然后尝试优化。

过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。 过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。过程中，你如果不是遇到非常严重的问题，禁止停下来问我，我希望你一次性完成这个任务。
上一个 commit 的代码不能保证可以运行成功，你直接调试就可以，如果 smem 你尝试各种方式都不够的话，就依次尝试尝试将 （1）写 gemm 的一部分放到 MMA 里（2）CE 和 MMA 进行合并。
