# `sG(row, col)` accumulator-mapped access

The hot `sG` reads in `kda_bwd_intra_kernel_sm90` do **not** walk shared memory in a simple row-major way. They follow the MMA accumulator fragment mapping:

```cpp
__device__ __forceinline__ void
get_acc_row_col(int tid, int v, int& row, int& col) {
    int lane = tid % 32;
    int warp_id = tid / 32;
    row = (lane / 4) + (v / 2) * 8;
    col = (lane % 4) * 2 + (v % 2) + warp_id * 8;
}
```

Equivalent form:

```cpp
struct AccPos {
    int row;
    int col;
};

__device__ __forceinline__ AccPos
get_acc_pos(int tid, int v) {
    int lane = tid & 31;
    int warp_id = tid >> 5;
    return {
        (lane >> 2) + ((v >> 1) << 3),
        ((lane & 3) << 1) + (v & 1) + (warp_id << 3),
    };
}
```

## Per-thread access pattern

- `tid = warp_id * 32 + lane`
- Each thread reads 4 positions: `v = 0..3`
- For fixed `tid`:
  - `v = 0, 1` access the same `row`
  - `v = 2, 3` access `row + 8`
  - `col` toggles between `base` and `base + 1`
  - `base = (lane % 4) * 2 + warp_id * 8`

Examples:

- `warp 0, lane 0`
  - `v=0 -> (0,0)`
  - `v=1 -> (0,1)`
  - `v=2 -> (8,0)`
  - `v=3 -> (8,1)`
- `warp 0, lane 7`
  - `v=0 -> (1,6)`
  - `v=1 -> (1,7)`
  - `v=2 -> (9,6)`
  - `v=3 -> (9,7)`
- `warp 1, lane 0`
  - `v=0 -> (0,8)`
  - `v=1 -> (0,9)`
  - `v=2 -> (8,8)`
  - `v=3 -> (8,9)`

So one warp covers an `8x8` subtile, and two warps together cover the full `16x16` accumulator tile. This is a scalar gather pattern derived from MMA accumulator layout, not a simple contiguous shared-memory scan.

## Why try padded stride for `sG`

If `sG` is stored as a plain padded row-major layout with stride `S`, a first-order bank model is:

```text
bank = (row * S + col) % 32
```

- `S = 32`: `bank = col`
  - different rows with the same `col` alias to the same bank
  - for the accumulator-mapped gather above, this can create heavy conflicts
- `S = 35`: `bank = (row * 3 + col) % 32`
  - each row shifts bank index by 3
  - `3` is coprime with `32`, so rows spread more evenly across banks

In practice, the kernel initializes `sG` with 16-byte `cp.async` copies:

```cpp
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dstG), "l"(&gG_tile(r, c)));
```

That means each row start should remain 16-byte aligned. So:

- stride 35 is a good bank-conflict thought experiment, but row stride = `35 * 4 = 140B`, which breaks 16-byte row alignment
- stride 36 keeps row stride = `36 * 4 = 144B`, so every row start is still 16-byte aligned

In a simple model of the `get_acc_row_col()` access set:

- stride 32 produced up to **8-way** bank conflicts
- stride 35 dropped that to about **2-way**
- stride 36 also stayed around **2-way** in the same simple model

This does **not** prove the current swizzled layout is worse in hardware. It explains why a padded layout is worth testing for the `sG(row, col)` hot path, and why stride 36 is the first alignment-safe candidate.
