# Bank Conflict Fix Spec: Shared Memory Swizzle for kda_bwd_intra_sm90

## 1. Problem Statement

NCU profiling shows **10.7-way average bank conflict** as the #1 performance bottleneck.
Current kernel achieves ~0.58x FLA Triton performance; bank conflicts are the dominant cause.

All shared memory layouts are plain row-major (`Stride<N, 1>`) with **no swizzle or padding**.
This causes severe bank conflicts on column-access patterns (same column, different rows).

## 2. Bank Conflict Root Cause Analysis

### Shared Memory Bank Architecture
- 32 banks, each 4 bytes wide, cycling every 128 bytes
- Bank number = `(byte_address / 4) % 32`
- Conflict: multiple threads in a warp access **different addresses** in the **same bank**

### Current Layouts and Conflict Severity

| Buffer       | Shape   | Type | Stride  | Row Bytes | Access Pattern         | Conflict |
|--------------|---------|------|---------|-----------|------------------------|----------|
| `sKG/sKBG`  | [32,16] | fp32 | [16,1]  | 64B       | MMA B-operand S2R      | **16-way** |
| `sDAqk/sDAkk`| [16,16]| fp32 | [16,1]  | 64B       | MMA A-operand S2R      | **8-way**  |
| `sG`         | [16,32] | fp32 | [32,1]  | 128B      | Element-wise scaling   | **16-way** |
| `sAcc`       | [16,32] | fp32 | [32,1]  | 128B      | db row-reduction       | **16-way** |
| `sQ/sK`      | [16,32] | bf16 | [32,1]  | 64B       | Element-wise read      | **8-way**  |

### Why Column Access Causes Conflicts

Example: `sG` with fp32 [16,32] stride [32,1]:
```
element(row, col) at byte offset = (row * 32 + col) * 4
bank = (row * 32 + col) % 32 = col % 32  (since 32 ≡ 0 mod 32)
```
All 16 rows with the same `col` map to the **same bank** → 16-way conflict.

Example: `sDAqk` with fp32 [16,16] stride [16,1]:
```
bank = (row * 16 + col) % 32
Row 0: bank = col
Row 2: bank = (32 + col) % 32 = col  → same as row 0!
```
Even rows (0,2,4,...) and odd rows (1,3,5,...) form two conflict groups → 8-way.

### Hottest Conflict Sites

1. **`gemm_m16n32k16` S2R copies** — called 2× per inner loop iteration
   - B operand (`sKG`): within each warp, 8 threads access same column → 8-way per warp
   - A operand (`sDAqk`): all 4 warps share A, threads access same column → 8-way
   - This is the **dominant** source of the 10.7x average

2. **Element-wise post-MMA scaling** — `exp2f(sG(row, col) - gn)` after every MMA
   - MMA accumulator mapping: `get_acc_row_col` puts 8 threads/warp on same col
   - sG bank = col % 32 → 8-way conflict per access

3. **Cooperative load loops** — building `sKG = k * exp2(gn - g)` from `sK`, `sG`
   - Sequential row access within `for (idx = tid; ...)` → less severe (2-way for bf16)

## 3. Solution: CuTe Swizzle

### Why Swizzle (not Padding)

| Approach | Pros | Cons |
|----------|------|------|
| **Padding** (+1 to stride) | Simple, effective | Wastes memory; incompatible with TMA |
| **Swizzle** (XOR remap) | Zero waste; TMA-compatible | Slightly more complex code |

We choose **swizzle** because:
- TMA buffers (sQ, sK, sG, sDAqk, sDAkk) require TMA-compatible layouts
- TMA hardware natively supports 32B/64B/128B swizzle modes
- CuTe `make_tma_copy` auto-detects swizzle from the smem layout
- Uniform approach: same technique for TMA and non-TMA buffers

### CuTe Swizzle Mechanics

`Swizzle<B, M, S>` transforms element index:
```
swizzled_idx = idx ^ ((idx >> S) & (((1 << B) - 1) << M))
```
This XORs `B` bits starting at position `M` with `B` bits starting at position `M + S`.

The key insight: for row-major layout with stride `N`, rows that differ put different bits
into positions `≥ log2(N)`. The swizzle XORs those high row-bits into the low column-bits,
ensuring different rows map to different banks.

### TMA Hardware Swizzle Modes

TMA supports three swizzle modes, each defined by which **byte address** bits are XORed:

| Mode | Byte bits XORed | Effective row width |
|------|-----------------|---------------------|
| 128B | bits [6:4] ^= bits [9:7] | 128 bytes/row |
| 64B  | bits [5:4] ^= bits [8:7] | 64 bytes/row  |
| 32B  | bit [4] ^= bit [7]       | 32 bytes/row  |

Translation to CuTe `Swizzle<B, M, S>` by element type:

| Type | Element size | 128B swizzle | 64B swizzle | 32B swizzle |
|------|-------------|--------------|-------------|-------------|
| fp32 | 4 bytes | `Swizzle<3,2,3>` | `Swizzle<2,2,3>` | `Swizzle<1,2,3>` |
| bf16 | 2 bytes | `Swizzle<3,3,3>` | `Swizzle<2,3,3>` | `Swizzle<1,3,3>` |

Derivation example (fp32, 128B swizzle):
- Byte addr = element_idx × 4
- Byte bits [6:4] = element bits [4:2]
- Byte bits [9:7] = element bits [7:5]
- XOR element bits [4:2] with [7:5] → `Swizzle<3, 2, 3>`

### Chosen Swizzle for Each Buffer

Match swizzle mode to row width:

| Buffer | Shape | Type | Row Bytes | Swizzle Mode | CuTe Swizzle |
|--------|-------|------|-----------|--------------|--------------|
| `sQ, sK` | [16,32] | bf16 | 64B | 64B | `Swizzle<2,3,3>` |
| `sG` | [16,32] | fp32 | 128B | 128B | `Swizzle<3,2,3>` |
| `sDAqk, sDAkk` | [16,16] | fp32 | 64B | 64B | `Swizzle<2,2,3>` |
| `sKG, sKBG` | [32,16] | fp32 | 64B | 64B | `Swizzle<2,2,3>` |
| `sAcc` | [16,32] | fp32 | 128B | 128B | `Swizzle<3,2,3>` |

### Conflict Reduction Verification

**Example: `sKG` [32,16] fp32 with `Swizzle<2,2,3>`**

Within one warp of the MMA B-operand S2R copy, 8 threads access column 0 (rows 0-7):
```
Original addresses:   0, 16, 32, 48, 64, 80, 96, 112
Original banks:       0,  0,  0,  0,  0,  0,  0,   0   → 8-way conflict

Swizzle: idx ^ ((idx >> 3) & 0xC)
Swizzled addresses:   0, 16, 36, 52, 72, 88, 108, 124
Swizzled banks:       0, 16,  4, 20,  8, 24,  12,  28  → 0 conflicts!
```

**Example: `sG` [16,32] fp32 with `Swizzle<3,2,3>`**

8 threads in a warp access column 0 (rows 0-7 via get_acc_row_col):
```
Original banks: all 0  → 8-way conflict

With Swizzle<3,2,3>: idx ^ ((idx >> 3) & 0x1C)
Row 0 (idx=0):   0 ^ 0  = 0,  bank 0
Row 1 (idx=32):  32 ^ 4  = 36, bank 4
Row 2 (idx=64):  64 ^ 8  = 72, bank 8
Row 3 (idx=96):  96 ^ 12 = 108, bank 12
Row 4 (idx=128): 128 ^ 16 = 144, bank 16
Row 5 (idx=160): 160 ^ 20 = 180, bank 20
Row 6 (idx=192): 192 ^ 24 = 216, bank 24
Row 7 (idx=224): 224 ^ 28 = 252, bank 28
→ 0 conflicts!
```

## 4. Implementation Plan

### 4.1 Define Swizzled Layouts (type aliases)

```cpp
// Swizzle functors
using Swz64B_f32  = Swizzle<2, 2, 3>;  // fp32, 64-byte row
using Swz128B_f32 = Swizzle<3, 2, 3>;  // fp32, 128-byte row
using Swz64B_bf16 = Swizzle<2, 3, 3>;  // bf16, 64-byte row

// Swizzled smem layouts
using SmemLayoutQK  = decltype(composition(Swz64B_bf16{},
    Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));      // [16,32] bf16
using SmemLayoutG   = decltype(composition(Swz128B_f32{},
    Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));      // [16,32] fp32
using SmemLayoutDA  = decltype(composition(Swz64B_f32{},
    Layout<Shape<Int<BC>, Int<BC>>, Stride<Int<BC>, _1>>{}));      // [16,16] fp32
using SmemLayoutB_op = decltype(composition(Swz64B_f32{},
    Layout<Shape<Int<BK>, Int<BC>>, Stride<Int<BC>, _1>>{}));      // [32,16] fp32
using SmemLayoutAcc = decltype(composition(Swz128B_f32{},
    Layout<Shape<Int<BC>, Int<BK>>, Stride<Int<BK>, _1>>{}));      // [16,32] fp32
```

### 4.2 TMA Descriptor Creation

`make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, swizzled_smem_layout)` automatically detects
the swizzle and configures the TMA descriptor's hardware swizzle mode.
No manual swizzle mode selection needed.

Use the same swizzled layout types for TMA smem layouts:
```cpp
using SmemLayoutQK_TMA = SmemLayoutQK;   // was separate, now unified
using SmemLayoutG_TMA  = SmemLayoutG;
using SmemLayoutDA_TMA = SmemLayoutDA;
```

### 4.3 SmemStorage Update

Array sizes use `cosize_v<SwizzledLayout>`. Swizzle is a bijection, so cosize
doesn't change (max index stays within the original range).

### 4.4 All Shared Memory Access Must Go Through Swizzled Tensor Views

**Critical**: every read/write to swizzled smem must use the swizzled layout.
Raw pointer indexing will produce wrong addresses.

```cpp
// Correct: access through swizzled tensor
auto sG = make_tensor(make_smem_ptr(smem.s_g.data()), SmemLayoutG{});
float val = sG(row, col);  // automatically applies swizzle

// WRONG: raw pointer arithmetic
float val = smem.s_g.data()[row * 32 + col];  // ignores swizzle!
```

Places requiring update:
- `gemm_m16n32k16`: create sA, sB with swizzled layouts ✓ (already uses type aliases)
- Cooperative load loops writing to `sKG`, `sKBG`: must write via swizzled tensor view
- Element-wise reads from `sG`, `sQ`, `sK`: already use tensor view ✓
- `sAcc` row-reduction: must use swizzled view for read/write
- `s_beta`, `s_gn`, `s_db`: these are 1D arrays, no swizzle needed

### 4.5 Non-TMA Cooperative Loads (sKG, sKBG)

These buffers are populated by cooperative loops. Change from raw index writes to
tensor view writes:

```cpp
// Before (wrong with swizzle):
sKG(n, k_dim) = kv * exp2f(...);  // This already uses tensor view, OK!

// The current code already writes via CuTe tensor — no change needed
// as long as SmemLayoutB_op is swizzled.
```

### 4.6 Phase 2 Cooperative Loads of dA (transposed gmem reads)

In Phase 2, dAqk/dAkk are loaded cooperatively (not TMA) with transposition:
```cpp
sDAqk(r, c) = dAqk_ptr[gmem_addr];  // Already uses tensor view
```
These automatically benefit from the swizzled `SmemLayoutDA`.

## 5. Expected Impact

| Source | Before | After | Improvement |
|--------|--------|-------|-------------|
| MMA B-operand (sKG) | 8-16 way | 0 | ~10x fewer stalls |
| MMA A-operand (sDA) | 8 way | 0-2 way | ~4x fewer stalls |
| sG element-wise | 8-16 way | 0-2 way | ~4-8x fewer stalls |
| sQ/sK element-wise | 4-8 way | 0-2 way | ~2-4x fewer stalls |
| sAcc reduction | 8-16 way | 0-2 way | ~4-8x fewer stalls |

Overall bank conflict should drop from ~10.7-way to ~1-2 way average.

## 6. Risks and Mitigations

1. **Correctness**: Swizzle is a bijection → no data loss. All accesses go through CuTe
   tensor views → addresses auto-remapped. Risk is low.

2. **TMA compatibility**: CuTe `make_tma_copy` validates swizzle patterns.
   If incompatible, it will fail at compile time (static_assert), not silently produce
   wrong results.

3. **gemm_m16n32k16 S2R copies**: The `make_tiled_copy_A/B` + `partition_S` path
   automatically uses the swizzled smem layout for address computation. No manual
   address math needed.

4. **cosize change**: Swizzle is a bijection on the index range, so cosize doesn't
   increase. No additional shared memory usage.
