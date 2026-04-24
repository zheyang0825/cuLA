# `chunk_kda_bwd_intra` 在 SM90 上的第二版技术报告

## 1. 背景

本文总结 `chunk_kda_bwd_intra` 在 SM90 上的**第二次实现尝试**。

第一次尝试整体上更接近 SM100 的结构：先用 **TMA** 搬运较大的 `64×64` tile，再做更偏 pipeline / warp specialization 的组织方式。这条路线在 SM90 上没有取得理想效果。一个核心原因是，这里的 backward-intra 子问题只能使用**同步的 SM80 风格 MMA**，而不是更适合做深流水重叠的异步 Hopper / Blackwell 风格张量核路径。这样一来，SM100 上那种 warp specialization 的收益很难真正落地；同时 occupancy 也提不上去，最终性能始终很难超过 FLA。

这并不意味着第一次尝试在方法论上完全错误，里面也可能夹杂了一些实现层面的技术问题；但至少在当前阶段，比较明确的判断是：

1. 如果 MMA 仍然是同步路径，那么 **SM100 那种 warp specialization 的结构在 SM90 上并不好用**。
2. 对这个 kernel 来说，**更小的 tile、更高的 occupancy、更轻的 block 结构**更符合实际。

基于这个判断，第二次尝试转向了更接近 FLA Triton 实现的拆分思路。

## 2. 第二版的总体设计

第二版不再让一个 thread block 负责大块计算，而是改成：

- 一个 thread block 只计算一个 **`16×32` 的梯度 tile**
- 这个 tile 对应一个 subchunk 的 **1/4**

这样设计的目标很明确：

- 简化控制流
- 提高 block 级并行度
- 降低单个 block 的状态规模
- 更贴合 SM90 上同步 SM80 风格 MMA 的使用方式

也就是说，第二版不再追求“结构上像 SM100”，而是追求“计算分解上更接近 FLA，但实现上用 CUTLASS/CuTe C++ 落地”。

## 3. 当前 kernel 的基本结构

当前实现可以概括为以下几部分：

1. **常驻 tile**
   - `Q / K / G` 先搬到 shared memory
   - `beta` 与 gate anchor（`s_gn`）也会准备好供后续复用

2. **Phase 1**
   - off-diagonal：处理更早 block 对 `dQ / dK` 的贡献
   - diagonal：处理下三角部分

3. **Phase 2**
   - off-diagonal：处理更晚 block 对转置 `dKT` 的贡献
   - diagonal：处理上三角部分

4. **最终 epilogue**
   - 写回 `dq / dk / dg / db`

从风格上说，这个 kernel 现在本质上仍然是一个**运行在 SM90 上的 SM80 风格 kernel**：

- 计算使用 **TF32 MMA**
- shared staging 主要依赖 **`cp.async`**
- 没有真正采用 Hopper 风格的异步 warp-specialized pipeline

### 3.1 一个 block 实际在算什么

一个 block 固定在当前 token 小块 `i` 上，负责生成一个 **`16 x 32`** 的输出梯度 tile。

输入张量形状可以写成：

```text
Q_i, K_i, G_i, DQ_i_prev, DK_i_prev, DG_i_prev : [16, 32]
beta_i                                            : [16]
```

输出张量形状为：

```text
DQ_i_out, DK_i_out, DG_i_out : [16, 32]
DB_i_out                     : [16]
```

块内只维护三个累计量：

```text
Delta_Q  : [16, 32]
Delta_K  : [16, 32]
Delta_KT : [16, 32]
```

它们的含义分别是：

- `Delta_Q`：当前 block 对 `dQ_i_out` 的累计量
- `Delta_K`：当前 block 对 `DK_i_out` 的 Phase 1 / diagonal 累计量
- `Delta_KT`：当前 block 对 `DK_i_out` 的 Phase 2 累计量

#### 来自更早小块 \(j<i\) 的贡献

对于每个更早小块 `j`，先构造：

```text
KG_j = K_j^T * exp2(G_i - G_j)   # shape [32, 16]
```

然后块内更新：

```text
Delta_Q += DA_qk_j * KG_j^T
Delta_K += DA_kk_j * KG_j^T
```

也就是说：**所有更早位置对当前 tile 的 `dQ` 和一部分 `dK` 贡献都累到 `Delta_Q`、`Delta_K` 里。**

#### 来自更晚小块 \(j>i\) 的贡献

对于每个更晚小块 `j`，先构造：

```text
QG_j  = Q_j^T * exp2(G_j - G_i_last)                    # shape [32, 16]
KBG_j = K_j^T * beta_j[:, None] * exp2(G_j - G_i_last) # shape [32, 16]
```

然后块内更新：

```text
Delta_KT += DA_qk_j * QG_j^T
Delta_KT += DA_kk_j * KBG_j^T
```

也就是说：**所有更晚位置通过转置路径传回来的 `dK` 贡献都累到 `Delta_KT` 里。**

#### 对角块

当 `j = i` 时，仍然做同类矩阵更新，只是在 `DA` 上施加三角 mask，因此 diagonal 的作用是补上当前 `16 x 16` 子块内部的合法项。

#### 最终输出

所有小块处理完后，block 用下面的公式生成最终输出：

```text
DQ_i_out = Delta_Q + DQ_i_prev
DK_i_out = Delta_K + Delta_KT + DK_i_diag + DK_i_prev
DG_i_out = Q_i * Delta_Q + K_i * (Delta_K - Delta_KT) + DG_i_prev
DB_i_out = RowReduce(Delta_K * K_i) + DB_i_prev
```

其中 `RowReduce` 表示对一个 `16 x 32` 张量按行求和，得到一个 `16` 维向量。

所以一句话概括就是：

> **一个 block 负责一个固定的 `16 x 32` 输出 tile，把所有更早块、更晚块和对角块对这个 tile 的贡献全部累起来，然后一次性写出 `dQ, dK, dG, db`。**

## 4. 内存搬运与实现观察

### 4.1 TMA 与 `cp.async`

这次工作中一个非常明确的观察是：对于当前这个 kernel 形状，**TMA 并不比 `cp.async` 更快**。

主要原因是这里每个 block 真正处理的 tile 很小。当工作单元收缩到 `16×32` 梯度 tile 之后，TMA 的 setup / shape 开销就很难被摊薄。因此当前最终实现采用的是 **`cp.async + shared memory`** 的方案，而不是 TMA。

### 4.2 MMA operand 的 shared-to-register 加载

当前 MMA 的 operand 加载方式如下：

- **A 矩阵**：使用 **`ldmatrix.x4`**
- **B 矩阵**：目前仍是 **标量 / LDS 风格 shared load**

这意味着 A 侧已经走到了更高效的 shared-to-register 路径，而 B 侧还没有。

因此一个很自然的后续问题是：

> B 矩阵有没有可能也改成 `ldmatrix.x4` 的加载方式？

这是后续值得继续研究的方向，但需要非常谨慎地处理：

- operand layout 的要求
- transpose 的关系
- register pressure
- shared memory 访问模式

## 5. 第二版中的关键优化迭代

第二版不是一次写完的，而是在 NCU 指导下持续做了若干轮小步迭代。

### 5.1 明显有效或者足够稳定的改动

1. **off-diagonal 热路径的全局内存向量化加载**
   - 对 off-diagonal 的 `Q/K/G` operand 构造改成了 4 路向量化加载
   - 这是一轮比较明确且有效的优化

2. **`dk_prev` 的最小化 old-gradient preload**
   - source counter 定位到 epilogue 里的 `gDk(row, col)` 是明显热点
   - 因此增加了一条最小化的 row-major preload：先把 `gDk_tile` 预取到 shared，再按 accumulator-mapped `(row, col)` 去读
   - 这个改动最重要的一点是：**没有把寄存器抬高，反而把寄存器压到了大约 47**

3. **`dq_prev` 的 preload 也做了尝试并暂时保留**
   - 它的收益不像 `dk` 那么明显
   - 但在当前测量下整体效果仍然可接受

### 5.2 没有带来净收益的尝试

1. **全局 `sG` layout / swizzle 调整**
   - 有些 swizzle 确实能改善一类 `sG` 访问的 bank conflict
   - 但同时也会把另一类 `sG` 访问搞坏
   - 最后整体体现出来的更多是副作用，而不是净收益

2. **大范围 epilogue preload / restaging**
   - 更重的 preload 思路往往会引入额外 shared staging、更多同步以及更高的 register pressure
   - 最终没有换来更好的性能

3. **`dA` shared staging 的 layout 重写**
   - 从理论上看，phase2 里的 transpose store 确实值得怀疑
   - 但一旦真的去动 `SmemLayoutDA` 的 base layout，就会直接跟 `ldmatrix` 的 source-layout 约束冲突
   - 在当前 TF32 + `ldmatrix.x4` 的设计下，这条路不容易安全吃到收益

## 6. 当前 profiling 的理解

从目前的 NCU 结果看，当前 kernel 依然主要受限于：

1. **shared memory 访问效率**
2. **部分 old-gradient 路径上的 uncoalesced / gather 型全局访问**

当前比较重要的 profiling 观察包括：

1. **Uncoalesced shared accesses 仍然很重**
2. **shared load bank conflict 通常比 shared store 更严重**
3. **source counter 已经明确指出：`dk_prev` 中 old-gradient gather 是非常值得优先处理的热点**

这件事很关键，因为它说明：

- 并不是所有 bank-conflict 警告都同样值得投入
- 也不是所有 shared layout 调整都能转化成真实收益

当前比较有效的工作方式应该是：

1. 看 NCU source counters
2. 找到真正最重的几行源码
3. 只做非常小、非常局部的改动
4. 一旦引入明显的寄存器上涨或者同步开销，就立刻停下

## 7. 为什么第二版是有希望的

虽然这个 kernel 还没有调完，但第二版明显比第一版更有希望：

1. 它更符合当前硬件和问题本身的实际约束。
2. 它不再依赖一个在同步 MMA 条件下很难做出收益的 SM100 风格重叠结构。
3. 它已经通过若干轮低风险优化，达到了一个可持续迭代的状态。
4. 当前 benchmark 已经在测试区间内稳定超过 FLA：

| total_len | FLA(ms) | cuLA(ms) | Speedup |
|---:|---:|---:|---:|
| 8192 | 4.7321 | 3.6831 | 1.28x |
| 16384 | 9.4490 | 7.3525 | 1.29x |
| 32768 | 18.3636 | 14.5386 | 1.26x |
| 65536 | 36.6823 | 29.0432 | 1.26x |
| 131072 | 73.3157 | 58.9597 | 1.24x |

这些结果说明第二版不仅方向正确，而且已经有了实用价值，后续继续抠热点仍然有意义。

## 8. 当前版本的限制

当前版本仍然有几个明显限制：

1. 它还不是一个真正 Hopper-native 的异步 kernel。
2. 它依然依赖 SM80 风格的 TF32 MMA 路线。
3. shared memory 访问模式还没有彻底清干净。
4. 某些理论上很有吸引力的优化，实际上会受到以下几方面的共同约束：
   - `ldmatrix` 对 source layout 的要求
   - transpose staging 的需求
   - register pressure
   - occupancy

## 9. 建议的后续方向

后续最合理的方式仍然是**继续按 NCU 逐点推进**：

1. 继续利用 source counters 清理剩余最大的 load / store 热点。
2. 继续评估 B operand 有没有办法向更 `ldmatrix` 风格的路径推进。
3. 对 `sG` 这种 mixed-access tensor，不要轻易做全局 layout 重写。
4. 优先选择以下类型的优化：
   - 不明显抬高寄存器
   - 不破坏 occupancy
   - 一次只针对一个热点
   - 可以用 source counter 直接验证

## 10. 总结

第二版的核心价值在于：它放弃了“SM90 必须结构上模仿 SM100”这个思路，转而采用了一个更小 tile、更高 occupancy、更接近 FLA 的设计方式。就目前为止，这个方向是正确的。

这一版最重要的实际结论是：

1. **对这种小 tile，TMA 并不天然优于 `cp.async`；这里最终是 `cp.async` 更合适。**
2. **如果 kernel 只能使用同步的 SM80 风格 MMA，那么 SM100 那种 warp specialization 很难真正发挥作用。**
3. **更小的 block 分解、更接近 FLA 的结构，是当前 SM90 上更现实的设计点。**
4. **基于 NCU 的小步优化已经开始产生效果，但必须严格控制副作用，尤其是寄存器和同步。**

在当前阶段，这一版已经是一个可用、可解释、可继续推进的实现；它不仅在测试中超过了 FLA，而且仍然保留了继续优化的空间。
