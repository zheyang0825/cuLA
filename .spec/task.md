我需要对 chunk_kda_bwd_intra 在 sm90 上进行优化，我需要使用 cutlass c++ 直接写这个 kernel，但是我之前做过一些尝试，首先对于 safe_gate 这种情况，在sm90 上只能使用 sm80.mma,
然后我发现warpgroup specialization 并不是很适用，因为 wgmma 用不起来，load 一个大的matrix意义不大，high occupy 感觉更实用一点，所以还是 fla triton 的实现看起来更符合预期，你参考一下
fla 的实现，并做 cutlass c++ 的设计，实现一个版本，再看怎么优化。


kda-backward-internal-feat-bwd_intra_opt 里面参考单测以及接口定义，其他内容都不要看，会有干扰。
third_party/flash-linear-attention/fla/ops/kda/chunk_bwd.py 是triton的参考对象。

tip:

尽量使用 ldmatrix4 指令 load s2r，但是因为 mma 计算使用 tf32，所以load fp32 的时候可以使用ldmatrix，但是 如果要 trans就不行，只能用 ld128，但是尽量组合使用，除非实现复杂度过于大了，先使用 ld128。
tma 可以用，不需要使用 cp.async,当然 cp.async 可作为备选。
swizzle 如果过于复杂也可以暂时不上，先写一个版本再慢慢处理 bank conflict

你参考 design.md ，不要看太多东西，分多步实现。


后续优化：
1. Swizzle - 减少 bank conflict
   目前加载 ADQK 和 ADKK 仍然是 2way bank conflict


2. KG 类似的计算
  当前直接从 GMEM 读取计算，太离谱了，给我改掉：
  cp. load K and G，
  （1）然后使用 LD128 和 LD64 加载到 register 中做计算
  或者（2）或者将 K 转成 fp32（使用额外的 SMEM），然后使用 ldmatrix.x4 加载512B，然后做计算。



3. ldmatrix
  MMA 计算 A 直接使用 ldmatrix 4 load的方式，tf32 相当于两个 bf16， A row major 可以直接用；
  但是 B 需要 trans ，就不太行，KG 本身也是写到 SMEM，能不能写的时候就 transpose


  参考 sm90_mma_array_tma_gmma_ss_warpspecialized.hpp:660-710 清除

          // Zero OOB rows (TMA may load next-sequence data for mid-sequence boundaries)
        if (is_boundary) {
            for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
                int r = idx / BK, c = idx % BK;
                if (i_ti + r >= T_seq) {
                    sQ(r, c) = __nv_bfloat16(0);
                    sK(r, c) = __nv_bfloat16(0);
                    sG(r, c) = 0.f;
                }
            }
        }