# CUTLASS C++ 实现 `chunk_kda_bwd_intra` (SM90, safe_gate=True)

## 目标

将 `fla/ops/kda/chunk_intra.py` 中的 `chunk_kda_bwd_kernel_intra`（safe_gate=True 路径）用 CUTLASS/cute C++ 重写，针对 SM90 (H100) 优化。

## 核心约束

- SM90 上使用 **sm80 MMA** (`mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32`)，不使用 wgmma
- **High occupancy** 策略，小 tile、多 CTA 并行
- 利用 SM90 的 **TMA (Tensor Memory Accelerator)** 加速 global→shared 数据搬运，cp.async 作为备选
- 数据类型：bf16/fp16 输入，fp32 累加

## 算法概要

Kernel 分两阶段，每阶段有 off-diagonal + diagonal 两部分：

**Phase 1 (计算 dq2, dk2)**：遍历当前 sub-chunk 之前的 sub-chunks 做 `dA @ (k * gate)` 累加，加上 diagonal 的 lower-triangular masked GEMM

**Phase 2 (计算 dkt)**：遍历当前 sub-chunk 之后的 sub-chunks 做 `dA.T @ (q * gate)` 累加，加上 diagonal 的 upper-triangular masked GEMM

**Epilogue**：`dg2 = q*dq2 + (dk2-dkt)*k + dg_in`，`dk2 = dk2 + dkt + dk_in`

## 架构设计

### Grid & CTA

```
Grid: (NK * NC, NT, B * HV)
  NK = ceil(K/BK), NC = ceil(BT/BC) = 4, NT = ceil(T/BT)
Block: 64 threads (2 warps)
```

### Tile 尺寸

| 参数 | 值 | 说明 |
|------|------|------|
| BC | 16 | sub-chunk 行数 |
| BK | 64 (主), 32 (高 occupancy 变体) | K 维度 tile |
| BT | 64 | chunk size |
| NC | 4 | BT/BC |

### MMA 配置 (tf32 精度)

Triton 在 SM90 上默认所有 `tl.dot` 使用 tf32 精度（`TRITON_F32_DEFAULT` 未设置时即 tf32）。`chunk_kda_bwd_kernel_intra` 中的 dot 也不例外。

```cpp
// tf32 MMA: m16n8k8, K_mma=8
using MMA_Atom_TF32 = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;

// 2 warps: 1 warp in M × 2 warps in N
// AtomLayoutMNK = Layout<Shape<_1, _2, _1>> → 64 threads total
using TiledMMA = TiledMMA<MMA_Atom_TF32, Layout<Shape<_1, _2, _1>>>;
// 结果: 每步 MMA tile = M=16, N=16, K=8
//   warp0 → N=[0,8), warp1 → N=[8,16)
//   对于 N=BK=64: 需要 64/16 = 4 步 N-iteration
//   对于 K=BC=16: 需要 16/8 = 2 步 K-reduction
```

**CUTLASS MMA Traits（来自 `mma_traits_sm80.hpp:162-177`）**：
```cpp
MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>:
  Shape_MNK = Shape<_16, _8, _8>
  ThrID     = Layout<_32>              // 1 warp = 32 threads
  ALayout   = Layout<Shape <Shape <_4,_8>, Shape <_2,_2>>,
                     Stride<Stride<_16,_1>,Stride<_8,_64>>>   // A: [M=16, K=8] tf32
  BLayout   = Layout<Shape <Shape <_4,_8>, _2>,
                     Stride<Stride<_8,_1>, _32>>               // B: [N=8, K=8] tf32
  CLayout   = SM80_16x8_Row                                    // C: [M=16, N=8] fp32
```

**操作数精度分析**：

Triton kernel 中 dot 的操作数实际都是 fp32：
- `dAqk`, `dAkk`: fp32（从 global 以 fp32 加载，或 fp32 累加结果）
- `b_kg = k * exp2(gn - g)`: fp32（exp2 运算结果）
- `b_qg = q * exp2(g - gn)`: fp32
- `b_k_exp = k * exp2(-g_diff)`: fp32

两个 MMA operand 都是 **fp32 → truncate to tf32 → mma.sync**。

**所有 dot 操作的维度映射**（safe_gate=True 路径）：

```
所有 dot 的形状统一为:  [BC, BC] @ [BC, BK] → [BC, BK]
即:  A[16, 16] @ B[16, BK] → C[16, BK]
MMA mapping:  M=16 (BC), K=16 (BC), N=BK (64 or 32)

Phase 1 off-diagonal (line 456-457):
  dq2 += dAqk[16,16] @ kg_j[16,BK]     // dA row-major
  dk2 += dAkk[16,16] @ kg_j[16,BK]

Phase 1 diagonal (line 495-496):
  dq2 += dA_masked[16,16] @ (k*exp)[16,BK]  // dA row-major, lower-tri masked
  dk2 += dA_masked[16,16] @ (k*exp)[16,BK]

Phase 2 off-diagonal (line 560-561):
  dkt += dAqk.T[16,16] @ qg_j[16,BK]   // dA 转置读
  dkt += dAkk.T[16,16] @ kbg_j[16,BK]

Phase 2 diagonal (line 598-599):
  dkt += dA_masked.T[16,16] @ (q*exp)[16,BK]  // dA 转置读, upper-tri masked
  dkt += dA_masked.T[16,16] @ (kb*exp)[16,BK]
```

### SMEM → Register 加载策略

**tf32 MMA 与 ldmatrix 的关系**：
- `ldmatrix.sync.aligned.x4.m8n8.shared.b16` 虽然以 16-bit 粒度搬运，但可用于加载 32-bit (fp32/tf32) 数据
- 原理：`[M, K]` fp32 重新解释为 `[M, 2K]` uint16，ldmatrix 把同行相邻的两个 uint16 放入同一 32-bit register，恰好是完整的 fp32 值
- **限制：`.trans` 变体不可用**——转置在 16-bit 粒度操作，会打碎 fp32 的高低半字。需要转置时走 `lds.128`

**A operand (dA) 加载**：
```
dA [BC, BC] = [16, 16] fp32 in SMEM
                 ↓ 按 K_mma=8 拆成 2 步
Step 0: [16, 8] fp32 = [16, 16] uint16 → ldmatrix.x4 (non-trans) → 4 regs/thread → tf32 MMA
Step 1: [16, 8] fp32 = [16, 16] uint16 → ldmatrix.x4 (non-trans) → 4 regs/thread → tf32 MMA
```
- ldmatrix.x4 加载 4×(8×8×2B) = 512B = 恰好是 [16,8] fp32 的大小
- 无需 `.trans`：dA 作为 A operand 不转置

**B operand (k\*gate) — 在 RF 中构造**：

B operand 需要 element-wise 计算后才能用于 MMA，因此不是直接从 SMEM ldmatrix 给 MMA，而是：
```
k: SMEM(bf16) → ldmatrix.x4 → RF(bf16) → bf16_to_fp32 → element-wise
g: SMEM(fp32) → lds.128     → RF(fp32) → exp2(gn - g) → element-wise
结果: b_kg = k_fp32 * exp2(gn - g)  // [BC, BK] fp32 在 RF 中
     → 直接作为 tf32 B operand 传入 MMA（fp32 截断为 tf32）
```
- 关键：ldmatrix 和 lds.128 负责高效地将 k/q/g 搬到 RF，但 MMA 的 B operand 是 RF 中计算得到的 fp32
- k/q 用 bf16 存储可节省 SMEM + 全局带宽，到 RF 再升精度
- B operand fragment 通过 `tiled_mma.partition_fragment_B()` 分配，手动填充 element-wise 结果

**Phase 2 转置读 dA.T**：
- dA 转置作为 A operand 时需要列方向读取
- ldmatrix.trans 不可用于 fp32 → 走 `lds.128` + 手动 register 排列
- 或者：先将 dA 转置到 SMEM 再用 ldmatrix.x4 正常读

**分场景汇总**：

| 数据 | 用途 | SMEM 格式 | SMEM→RF 方式 | 说明 |
|------|------|----------|-------------|------|
| dA (non-trans) | MMA A operand | fp32 | `ldmatrix.x4` (b16 reinterpret) | [16,8] fp32 = [16,16] uint16 |
| dA.T (trans) | MMA A operand | fp32 | `lds.128` | .trans 不可用于 fp32 |
| k, q | element-wise → MMA B | bf16 | `ldmatrix.x4` | 原生 16-bit，加载后升精度 |
| g (gate) | element-wise 计算 | fp32 | `lds.128` | exp2 参数 |
| beta | scalar 乘法 | fp32 | 直接 load | 小数据，不走 MMA |

### SMEM → RF 的 cute Copy Atom

```cpp
// dA (fp32 via b16 reinterpret): ldmatrix non-trans
// 关键：SMEM layout 必须让 fp32 的高低 uint16 相邻，ldmatrix 自然将它们放入同一 register
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>;

// dA.T (fp32 需要转置): 回退到 vectorized load
using SmemCopyAtomA_Trans = Copy_Atom<UniversalCopy<uint128_t>, float>;

// k/q (bf16): ldmatrix 原生加载
using SmemCopyAtomB_bf16 = Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t>;
using SmemCopyAtomB_fp16 = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;

// g (fp32): vectorized load
using SmemCopyAtomG = Copy_Atom<UniversalCopy<uint128_t>, float>;
```

**SMEM Swizzle 推导**：

Swizzle<B, M, S> 的含义：对地址的 bit[S+B-1:S] 与 bit[S+B+M-1:S+B] 做 XOR。
目的：消除 bank conflict（SMEM 32 banks，每 bank 4B，连续 128B = 32 banks）。

```
dA [16, 16] fp32: 每行 16*4B = 64B = 16 banks
  行跨度 64B，需要 swizzle 让不同行同列访问不冲突
  ldmatrix 以 b16 reinterpret: [16, 32] uint16, 每行 64B
  对 uint16 layout: Swizzle<3, 0, 3>
    B=3: XOR 3 bits, S=3: 从 bit3 开始, M=0
    这意味着 row_idx 的低 3 bit 与 col 的 bit[5:3] XOR
    每个 ldmatrix 8x8 tile 的 8 行分散到不同 bank
  对 fp32 解释: 等价于 Swizzle<2, 0, 2>（因为 fp32 是 4B，bank 粒度变为 1 element）

k/q [16, BK=64] bf16: 每行 64*2B = 128B = 32 banks (刚好满 bank)
  Swizzle<3, 0, 3>: XOR row[2:0] with col[5:3]
  适合 ldmatrix: 8 行的 128-bit 段不会 bank conflict

g [16, BK=64] fp32: 每行 64*4B = 256B = 64 banks (超过 32，自然 wrap)
  Swizzle<3, 0, 3>: XOR row[2:0] with col[5:3] (以 fp32 为粒度)
  lds.128 加载：每线程 4 个 fp32 = 16B，32 线程并行
```

```cpp
// dA [16, 16] fp32，按 b16 reinterpret 为 [16, 32] uint16 做 swizzle
// ldmatrix.x4 需要 128-bit 对齐 + 行间无 bank conflict
using SmemLayoutDA = decltype(composition(
    Swizzle<3, 0, 3>{},
    Layout<Shape<_16, _32>, Stride<_32, _1>>{}  // [16, 32] uint16 = [16, 16] fp32
));

// k/q [16, 64] bf16
using SmemLayoutKQ = decltype(composition(
    Swizzle<3, 0, 3>{},
    Layout<Shape<_16, _64>, Stride<_64, _1>>{}
));

// g [16, 64] fp32
using SmemLayoutG = decltype(composition(
    Swizzle<3, 0, 3>{},
    Layout<Shape<_16, _64>, Stride<_64, _1>>{}
));
```

### Shared Memory Layout

**设计原则**：
1. Phase 1 和 Phase 2 的 off-diagonal loop 使用 double buffer
2. 当前 sub-chunk 的 q/k/g/beta 数据在 kernel 开始时加载一次，持续复用
3. Phase 2 需要额外的 q_j/k_j/g_j/beta_j 数据（来自后续 sub-chunks），复用 Phase 1 的 double buffer

**BK=64 时**（主配置）：
```
═══════════════════════════════════════════════════════════════
 持久区域（kernel 全程保持）:
   s_q       [16, 64] bf16     2,048 B    // 当前 sub-chunk 的 q
   s_k       [16, 64] bf16     2,048 B    // 当前 sub-chunk 的 k
   s_g       [16, 64] fp32     4,096 B    // 当前 sub-chunk 的 g
   s_beta    [16]     fp32        64 B    // 当前 sub-chunk 的 beta
   s_gn      [64]     fp32       256 B    // anchor gate 值
                            ─────────────
                   持久区小计:  8,512 B ≈ 8.3 KB

 Double-buffer 区域（off-diagonal loop 用，2 stages）:
   s_dA_qk   [16, 16] fp32     1,024 B × 2 = 2,048 B
   s_dA_kk   [16, 16] fp32     1,024 B × 2 = 2,048 B
   s_k_j     [16, 64] bf16     2,048 B × 2 = 4,096 B   // 远端 k 或 q
   s_g_j     [16, 64] fp32     4,096 B × 2 = 8,192 B   // 远端 g
                            ─────────────
              Double-buffer 小计: 16,384 B = 16 KB

 Phase 2 附加（复用 s_k_j/s_g_j 空间）:
   s_beta_j  [16]     fp32        64 B × 2 = 128 B      // 远端 beta
                            ─────────────
═══════════════════════════════════════════════════════════════
 Total SMEM: ~24.9 KB（远低于 SM90 的 228 KB 限额）
```

**BK=32 时**（高 occupancy 变体）：
```
持久区: 2×(16×32×2B) + 16×32×4B + 64B + 128B = 2×1KB + 2KB + 0.2KB ≈ 4.2 KB
Double-buffer: 2×(1KB+1KB+1KB+2KB) = 10 KB
Total: ~14.2 KB
```

使用 cute::Swizzle 避免 bank conflict（参见上面的推导）。

### 数据搬运：TMA vs cp.async

**TMA 适用的访问模式**（SM90 特有，完全异步，不占 warp 执行资源）：

| 数据 | 形状 | 全局 stride | TMA 适用性 |
|------|------|-------------|-----------|
| q, k | [BC, BK] | (H\*K, 1) | TMA 2D，stride 规整 |
| g (fp32) | [BC, BK] | (HV\*K, 1) | TMA 2D，但 fp32 需要 `SM90_TMA_LOAD` 支持 |
| dAqk, dAkk | [BC, BC] | (HV\*BT, 1) | TMA 2D，stride 较大但可行 |
| dA 转置读 | [BC, BC] | (1, HV\*BT) | TMA 2D 列读，stride 跨度大，L2 利用率低 |
| beta | [BC] | (HV,) | TMA 1D 或直接 load |

**TMA 优势**：
- 完全异步：load 不占 warp 寄存器和执行单元，由 TMA unit 独立完成
- Barrier 驱动：配合 `mbarrier` 实现更精细的 load/compute overlap
- 自动 OOB 填零：无需手写 boundary predicate
- 对于小 tile（16×64）也有效，因为省下的 warp cycles 可用于 element-wise 计算

**TMA 实现要点**：
- Host 端需要创建 `TmaDescriptor`（通过 `cute::make_tma_copy`）
- 每个不同 stride 模式需要独立的 TMA descriptor
- Descriptor 通过 kernel 参数或 constant memory 传入
- 只需 1 个线程发起 TMA load，其余线程等 barrier

**流水线设计**（TMA + mbarrier double buffering）：

```
Stage 0: TMA.load(smem_ping, ...) → arrive(bar0)
                                              ↓
Stage 1: TMA.load(smem_pong, ...) → arrive(bar1)   wait(bar0) → Compute[0]
                                              ↓                      ↓
Stage 0: TMA.load(smem_ping, ...) → arrive(bar0)   wait(bar1) → Compute[1]
```

NC=4 时最多 3 次 off-diagonal 迭代，流水线浅但 TMA 的异步特性意味着即使 1 次迭代也能受益（load 与 diagonal compute 重叠）。

**cp.async 作为备选**：对于 g (fp32) 数据，如果 TMA fp32 路径有问题，可回退到 `cp.async.bulk` 或显式 load + `__syncthreads`。

### Register Budget

| BK | Regs/thread | CTAs/SM | Occupancy |
|----|-------------|---------|-----------|
| 64 | ~170 | 6 | ~37% |
| 32 | ~90 | 11 | ~69% |

## Kernel 伪代码（CUTLASS C++ 映射）

```cpp
__global__ void chunk_kda_bwd_intra_kernel(/* params */) {
    // ===== Grid mapping =====
    int i_kc = blockIdx.x;       // NK * NC
    int i_t  = blockIdx.y;       // NT
    int i_bh = blockIdx.z;       // B * HV
    int i_k = i_kc / NC, i_i = i_kc % NC;
    int i_b = i_bh / HV, i_hv = i_bh % HV;
    int i_h = i_hv / (HV / H);

    // 计算偏移 + varlen 处理（略）
    int i_ti = i_t * BT + i_i * BC;
    if (i_ti >= T) return;

    // ===== 分配 RF accumulators =====
    // cute fragment: b_dq2[BC, BK], b_dk2[BC, BK] — fp32 累加器
    auto b_dq2 = make_tensor<float>(partition_shape_C(tiled_mma, Shape<_BC, _BK>{}));
    auto b_dk2 = make_tensor<float>(partition_shape_C(tiled_mma, Shape<_BC, _BK>{}));
    clear(b_dq2); clear(b_dk2);

    // ===== 加载当前 sub-chunk 数据到 SMEM =====
    // TMA/cp.async: q[i_ti : i_ti+BC, i_k*BK : (i_k+1)*BK] → s_q
    //              k[i_ti : i_ti+BC, i_k*BK : (i_k+1)*BK] → s_k
    //              g[i_ti : i_ti+BC, i_k*BK : (i_k+1)*BK] → s_g
    //              beta[i_ti : i_ti+BC] → s_beta
    //              g[i_ti + BC//2, i_k*BK : (i_k+1)*BK] → s_gn  (anchor)
    __syncthreads();

    // ===== Phase 1: Off-diagonal (i_j < i_i) =====
    if (i_i > 0) {
        // b_gn = g[i_ti, :] — 第一行作为 off-diagonal anchor
        for (int i_j = 0; i_j < i_i; i_j++) {
            // --- Load to SMEM (double buffer) ---
            // dAqk[i_ti, i_j*BC : (i_j+1)*BC] → s_dA_qk   // [BC, BC] fp32
            // dAkk[i_ti, i_j*BC : (i_j+1)*BC] → s_dA_kk
            // k[i_t*BT + i_j*BC, i_k*BK]      → s_k_j     // [BC, BK] bf16
            // g[i_t*BT + i_j*BC, i_k*BK]      → s_g_j     // [BC, BK] fp32
            __syncthreads();  // or TMA barrier wait

            // --- Construct B operand in RF ---
            // b_kg_j = k_j * exp2(gn - g_j)   // [BC, BK] fp32
            // 1. ldmatrix s_k_j → RF(bf16) → cvt fp32
            // 2. lds.128 s_g_j → RF(fp32) → exp2(gn - g_j)
            // 3. b_kg_j = k_fp32 * gate_exp

            // --- MMA: dA @ kg → accumulate ---
            // A operand: ldmatrix.x4 s_dA_qk → [16,8] fp32 per MMA step
            // B operand: b_kg_j (RF fp32)
            // b_dq2 += dAqk @ kg_j   (4 N-iterations × 2 K-reductions)
            // b_dk2 += dAkk @ kg_j
        }
        // b_dq2 *= exp2(g_curr - gn)
        // b_dk2 *= exp2(g_curr - gn)
    }

    // ===== Phase 1: Diagonal (safe_gate) =====
    // 加载 diagonal dA 块: dAqk[i_ti, i_i*BC : (i_i+1)*BC]
    // 应用 lower-triangular mask: m_i_diag_qk = (row >= col) & boundary
    // b_gn_diag = g[i_ti + BC//2, :]  (midpoint anchor)
    //
    // b_k_exp = k * exp2(-(g - gn_diag))
    // b_dAqk_masked = tl.where(lower_tri, dAqk, 0)
    // b_dq2 += dAqk_masked @ b_k_exp * exp2(g - gn_diag)
    // b_dk2 += dAkk_masked @ b_k_exp * exp2(g - gn_diag)
    //
    // 注意：diagonal 的 mask 需要在 MMA 之前应用到 dA（SMEM 中 mask），
    //       或者在 MMA 之后用 element-wise mask 修正
    //       推荐：MMA 前在 RF 中对 A fragment 做 mask

    // ===== Phase 1 Epilogue =====
    // b_db = row_sum(b_dk2 * k)   // [BC] 标量
    // b_dk2 *= beta                // [BC, BK]
    // b_dg2 = q * b_dq2           // [BC, BK] 部分 dg
    // b_dq2 += load(dq_upstream)  // 加上 upstream dq
    // store b_dq2 → dq2
    // store b_db → db
    __syncthreads();  // barrier between Phase 1 and Phase 2

    // ===== Phase 2: Off-diagonal (i_j > i_i) =====
    auto b_dkt = make_tensor<float>(...);
    clear(b_dkt);

    int NC_actual = min(NC, cdiv(T - i_t * BT, BC));
    if (i_i < NC_actual - 1) {
        // b_gn2 = g[min(i_ti+BC, T) - 1, :]  // 末行 anchor
        for (int i_j = i_i + 1; i_j < NC_actual; i_j++) {
            // --- Load to SMEM ---
            // q[i_t*BT+i_j*BC, :] → s_k_j (复用 buffer, 此处存 q)
            // k[i_t*BT+i_j*BC, :] → 也需要（用于 kb*g 计算）
            // g[i_t*BT+i_j*BC, :] → s_g_j
            // beta[i_t*BT+i_j*BC] → s_beta_j
            // dAqk 转置读: dAqk[(i_i*BC, i_t*BT+i_j*BC), col_stride=BT, row_stride=1]
            //   即 stride=(1, HV*BT)，转置 [BC, BC]

            // --- Construct B operand in RF ---
            // b_gkn = exp2(g_j - gn2)
            // b_qg = q_j * where(m_j, b_gkn, 0)     // [BC, BK]
            // b_kbg = k_j * beta_j * where(m_j, b_gkn, 0)

            // --- MMA: dA.T @ B → accumulate ---
            // A operand: dA.T from SMEM (lds.128, 因为 .trans 不可用)
            // b_dkt += dAqk.T @ qg   // 4 N-iter × 2 K-red
            // b_dkt += dAkk.T @ kbg
        }
        // b_dkt *= exp2(gn2 - g_curr)
    }

    // ===== Phase 2: Diagonal (safe_gate) =====
    // 加载 diagonal dA 转置: stride=(1, HV*BT)
    // 应用 upper-triangular mask: m_i_diag_kk = (row <= col) & boundary
    // b_gn_diag = g[i_ti + BC//2, :]
    //
    // b_q_exp = q * exp2(g - gn_diag)
    // b_kb_exp = k * beta * exp2(g - gn_diag)
    // b_dkt += dAqk_masked.T @ b_q_exp * exp2(-(g - gn_diag))
    // b_dkt += dAkk_masked.T @ b_kb_exp * exp2(-(g - gn_diag))

    // ===== Phase 2 Epilogue =====
    // b_dg2 += (b_dk2_phase1 - b_dkt) * k + load(dg_upstream)
    // b_dk2_final = b_dk2_phase1 + load(dk_upstream) + b_dkt
    // store b_dk2_final → dk2
    // store b_dg2 → dg2
}
```

**MMA 执行细节**（以 Phase 1 off-diagonal 一次 `dAqk @ kg_j` 为例）：

```
输入:
  A = dAqk [BC=16, BC=16] fp32 (in SMEM)
  B = kg_j [BC=16, BK=64] fp32 (in RF, element-wise 计算得到)
  C = b_dq2 [BC=16, BK=64] fp32 (accumulator in RF)

TiledMMA config: atom m16n8k8, AtomLayout=<1,2,1>
  → per-step tile: M=16, N=16, K=8

执行循环:
  for n = 0..3:     // N=64 / 16 = 4 步
    for k = 0..1:   // K=16 / 8 = 2 步 (K-reduction)
      A_frag = ldmatrix.x4 s_dA[*, k*8:(k+1)*8]  // [16,8] fp32 via b16 reinterpret
      B_frag = b_kg_j[k*8:(k+1)*8, n*16:(n+1)*16]  // 已在 RF 中
      mma.sync C_frag += A_frag * B_frag           // m16n8k8 × 2 warps

总 MMA 指令: 4(N) × 2(K) × 1(M) = 8 次 mma.sync per dot
每个 CTA 2 个 dot (dAqk + dAkk) = 16 次 mma.sync per iteration
```

## 实现步骤

### Step 1: 创建项目文件结构

```
fla/ops/kda/csrc/
├── chunk_kda_bwd_intra.cuh    # kernel 实现
├── chunk_kda_bwd_intra.cu     # host dispatch + template instantiation
└── chunk_kda_bwd_intra.h      # 公共头文件/接口声明
```

### Step 2: 实现 kernel 主体 (`chunk_kda_bwd_intra.cuh`)

1. Template 参数化：
   ```cpp
   template <int BC_, int BK_, int BT_, int NC_, typename Element>
   // Element = cute::bfloat16_t 或 cute::half_t
   // BC_=16, BT_=64, NC_=4
   // BK_=64 或 32
   ```

2. 类型定义：
   ```cpp
   using MMA_Atom_TF32 = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
   using TiledMMA = TiledMMA<MMA_Atom_TF32, Layout<Shape<_1, _2, _1>>>;
   // 64 threads, 2 warps
   ```

3. SMEM 布局定义（含 swizzle）
4. Global→SMEM copy 定义（TMA 或 cp.async）
5. Phase 1: off-diagonal loop + diagonal safe_gate GEMM
   - off-diagonal: 对 i_j in [0, i_i) 循环
   - diagonal: lower-triangular masked MMA（mask 应用在 A fragment 上）
6. Phase 1 epilogue:
   - `b_db = row_reduce(b_dk2 * s_k)`（需要 warp-level reduction）
   - `b_dk2 *= beta`
   - `b_dg2 = s_q * b_dq2`（保留给 Phase 2 epilogue）
   - `b_dq2 += load(dq_upstream)`
   - store `dq2`, `db`
7. `__syncthreads()` barrier（Phase 1 存的 dq2 可能被后续读取）
8. Phase 2: off-diagonal loop + diagonal safe_gate GEMM
   - off-diagonal: 对 i_j in (i_i, NC) 循环
   - diagonal: upper-triangular masked MMA
9. Phase 2 epilogue:
   - `b_dg2 += (b_dk2_saved - b_dkt) * s_k + load(dg_upstream)`
   - `b_dk2_final = b_dk2_saved + load(dk_upstream) + b_dkt`
   - store `dk2`, `dg2`

**Phase 2 SMEM 需求分析**：Phase 2 off-diagonal 每次迭代需要加载 5 种数据：
  - dAqk.T [BC,BC] fp32: 需要转置读（stride=(1, HV*BT)）
  - dAkk.T [BC,BC] fp32: 同上
  - q_j [BC,BK] bf16
  - k_j [BC,BK] bf16 + beta_j [BC] fp32（用于计算 kb）
  - g_j [BC,BK] fp32

问题：Phase 1 的 s_k_j 只有 1 个 bf16 buffer（per stage），但 Phase 2 需要 q_j 和 k_j 两个。
解决方案：
  - **方案 A**：Phase 2 的 s_k_j 分为两半，分别存 q 和 k（但 BK=64 时 q_j+k_j = 4KB bf16，刚好等于 2 个 s_k_j stage）
  - **方案 B**：Phase 2 每次迭代先加载 q_j 做一个 MMA，再加载 k_j 做另一个 MMA（序列化，但简化 SMEM）
  - **方案 C**：增加 SMEM（再加一个 bf16 buffer），总 SMEM 仍在预算内
  - **推荐方案 B**：与 Triton 实现一致（Triton 也是分别加载 q 和 k）

### Step 3: Host dispatch (`chunk_kda_bwd_intra.cu`)

1. 根据 K 选择 BK=32 或 BK=64
2. 根据 dtype 选择 bf16/fp16 模板实例
3. 计算 grid/block/smem 大小
4. Launch kernel
5. Post-process: `db = db2.sum(dim=0) + db_upstream`

### Step 4: PyTorch binding

1. `torch::Tensor` 接口封装
2. 替换 `chunk_kda_bwd_intra` 调用路径（条件性：SM90 + safe_gate 时使用 CUTLASS kernel）

## 性能瓶颈分析

**计算量**（BK=64, BC=16, safe_gate=True, per CTA）：

```
Phase 1 off-diagonal: i_i 次迭代 (最多 3 次)
  每次: 2 个 MMA GEMM [16,16] @ [16,64] = 2 × 16×64×16 = 32,768 FLOPs
  最大: 3 × 32,768 = 98,304 FLOPs

Phase 1 diagonal:
  2 个 masked MMA [16,16] @ [16,64] ≈ 32,768 FLOPs (含 mask 开销)

Phase 2 off-diagonal: (NC-1-i_i) 次迭代 (最多 3 次)
  每次: 4 个 MMA (dAqk@q + dAkk@kb × 转置) = 4 × 16,384 = 65,536 FLOPs
  最大: 3 × 65,536 = 196,608 FLOPs

Phase 2 diagonal:
  4 个 masked MMA ≈ 65,536 FLOPs

Element-wise ops: exp2, mul, mask, reduce ≈ ~20% overhead

总计: ~393K FLOPs per CTA (worst case, i_i=0 或 i_i=3)
```

**内存访问量**（BK=64, per CTA, worst case）：
```
Global reads:
  q,k [16,64] bf16: 2 × 2KB = 4KB (一次性)
  g [16,64] fp32: 4KB (一次性)
  beta [16] fp32: 64B (一次性)
  gn [64] fp32: 256B
  off-diagonal 每次: dA×2 [16,16] fp32 = 2KB, k/q [16,64] bf16 = 2KB, g [16,64] fp32 = 4KB = 8KB
  最多 6 次 off-diag (Phase1+Phase2) × 8KB = 48KB
  diagonal: dA×2 [16,16] fp32 = 2KB × 2 phases = 4KB
  upstream dq/dk/dg: 3 × [16,64] fp32 = 12KB

Global writes:
  dq2, dk2, dg2: 3 × [16,64] fp32 = 12KB
  db: [16] fp32 = 64B

Total: ~80 KB global memory per CTA
```

**Arithmetic Intensity**: ~393K / 80KB ≈ 4.8 FLOPs/Byte — **memory-bound**

这说明：
1. **数据搬运优化**（TMA/cp.async pipeline）比 MMA 配置更关键
2. **Double buffering** 对隐藏 global latency 至关重要
3. **BK=32** 可能更优：减少一半 element-wise 计算的数据量，虽然 MMA 效率相同
4. SMEM 足够小可以高 occupancy，有助于隐藏 global latency

## 关键参考文件

- `fla/ops/kda/chunk_intra.py:367-630` — `chunk_kda_bwd_kernel_intra` Triton 实现
- `fla/ops/kda/chunk_intra.py:849-914` — `chunk_kda_bwd_intra` Python wrapper
- `fla/ops/kda/chunk_bwd.py:558-573` — 调用点，定义输入输出张量
- `fla/ops/kda/naive.py` — 数学参考实现
