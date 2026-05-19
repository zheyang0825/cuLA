// Test: verify wgmma m64n32k8 RS (A in registers, B in shared memory)
// Computes D[64x32] = A[64x8] @ B[8x32] using wgmma
// A = all 1.0 (in registers), B = identity-like (in shared memory)
// Expected: D = A @ B = row sums of B
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

// Helper: convert smem pointer to uint32 address
__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr;\n"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}

constexpr uint32_t TF32_MASK = 0xFFFFE000u;

// Construct a B matrix descriptor for wgmma
// B is [8][32] tf32 in shared memory, stored row-major (128 bytes per row)
__device__ __forceinline__ uint64_t make_b_desc(void const* smem_ptr) {
    uint32_t addr = smem_ptr_to_uint(smem_ptr);
    uint64_t desc = 0;
    // bits [0:13]: start_address >> 4
    desc |= ((uint64_t)(addr >> 4)) & 0x3FFF;
    // bits [16:29]: leading_byte_offset >> 4
    // For row-major B[8][32] tf32: leading dim stride = 32*4 = 128 bytes -> 128/16 = 8
    desc |= ((uint64_t)(8 & 0x3FFF)) << 16;
    // bits [32:45]: stride_byte_offset >> 4
    // stride between consecutive "core matrices" (8-col groups): 8*4 = 32 bytes -> 32/16 = 2
    desc |= ((uint64_t)(2 & 0x3FFF)) << 32;
    // bits [48:50]: base_offset = 0
    // bits [56:57]: layout_type = 0 (no swizzle)
    return desc;
}

// wgmma m64n32k8 RS TN: A[64x8] in regs, B[8x32] in smem, D[64x32] in regs
__device__ __forceinline__ void wgmma_m64n32k8_rs(
    float d[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint64_t desc_b, int scale_d)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %20, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %21,"
        " p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(scale_d), "l"(desc_b)
    );
}

// Test kernel: 128 threads = 1 warpgroup
// A = all 1.0, B = known pattern → D = A @ B
__global__ void __launch_bounds__(128)
test_wgmma_kernel(float *D_out, const float *B_in) {
    __shared__ float B_smem[8][32];  // row-major

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int gid = lane / 4;
    int tid_in_grp = lane % 4;

    // Load B into shared memory
    for (int idx = tid; idx < 8 * 32; idx += 128) {
        int r = idx / 32, c = idx % 32;
        B_smem[r][c] = B_in[r * 32 + c];
    }
    __syncthreads();

    // Set A = all 1.0 (TF32 masked)
    uint32_t a0 = __float_as_uint(1.0f) & TF32_MASK;
    uint32_t a1 = a0;
    uint32_t a2 = a0;
    uint32_t a3 = a0;

    // Zero-init D
    float d[16] = {};

    // Construct B descriptor
    uint64_t desc_b = make_b_desc(&B_smem[0][0]);

    // Issue wgmma
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_m64n32k8_rs(d, a0, a1, a2, a3, desc_b, 1);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    // Write D to global memory
    // D fragment layout for m64n32k8:
    // row0 = warp_id*16 + gid, row1 = row0 + 8
    int row0 = warp_id * 16 + gid;
    int row1 = row0 + 8;
    // Columns: nb=0..3 maps to d[4*nb:4*nb+3]
    //   d[4*nb+0] = D[row0, nb*8 + 2*tid]
    //   d[4*nb+1] = D[row0, nb*8 + 2*tid+1]
    //   d[4*nb+2] = D[row1, nb*8 + 2*tid]
    //   d[4*nb+3] = D[row1, nb*8 + 2*tid+1]
    for (int nb = 0; nb < 4; nb++) {
        int col0 = nb * 8 + 2 * tid_in_grp;
        int col1 = col0 + 1;
        D_out[row0 * 32 + col0] = d[4*nb + 0];
        D_out[row0 * 32 + col1] = d[4*nb + 1];
        D_out[row1 * 32 + col0] = d[4*nb + 2];
        D_out[row1 * 32 + col1] = d[4*nb + 3];
    }
}

int main() {
    // B = identity-like: B[k][n] = (k == n % 8) ? 1.0 : 0.0
    // So each row k has 1s at columns k, k+8, k+16, k+24
    // D = A @ B where A is all-1s [64x8]:
    // D[m][n] = sum_k A[m][k] * B[k][n] = sum_k 1 * B[k][n] = col_sum(B)[n]
    // B col_sum[n] = 1.0 (each column has exactly one 1)
    // So D should be all 1.0

    float h_B[8 * 32] = {};
    for (int k = 0; k < 8; k++) {
        for (int n = 0; n < 32; n++) {
            h_B[k * 32 + n] = (k == (n % 8)) ? 1.0f : 0.0f;
        }
    }

    float *d_B, *d_D;
    cudaMalloc(&d_B, 8 * 32 * sizeof(float));
    cudaMalloc(&d_D, 64 * 32 * sizeof(float));
    cudaMemcpy(d_B, h_B, 8 * 32 * sizeof(float), cudaMemcpyHostToDevice);

    test_wgmma_kernel<<<1, 128>>>(d_D, d_B);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));

        // Try different descriptor formats
        printf("Trying alternative descriptors...\n");
        return 1;
    }

    float h_D[64 * 32];
    cudaMemcpy(h_D, d_D, 64 * 32 * sizeof(float), cudaMemcpyDeviceToHost);

    // Check results
    int errors = 0;
    for (int m = 0; m < 64; m++) {
        for (int n = 0; n < 32; n++) {
            float expected = 1.0f;
            float got = h_D[m * 32 + n];
            if (fabsf(got - expected) > 0.01f) {
                if (errors < 10) {
                    printf("MISMATCH D[%d][%d]: expected %.4f, got %.4f\n", m, n, expected, got);
                }
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("ALL CORRECT! D = A @ B matches expected values.\n");
    } else {
        printf("%d / %d mismatches\n", errors, 64*32);
        // Print first few rows
        printf("\nD[0,:8] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", h_D[n]);
        printf("\nD[1,:8] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", h_D[32 + n]);
        printf("\nD[8,:8] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", h_D[8 * 32 + n]);
        printf("\nD[16,:8] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", h_D[16 * 32 + n]);
        printf("\n");
    }

    // Test 2: B = all 1s, A = different per row
    // D[m][n] = sum_k A[m][k] * 1 = sum of A[m] = 8 (since A is all 1s)
    printf("\n=== Test 2: B = all-ones ===\n");
    float h_B2[8 * 32];
    for (int i = 0; i < 8 * 32; i++) h_B2[i] = 1.0f;
    cudaMemcpy(d_B, h_B2, 8 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    test_wgmma_kernel<<<1, 128>>>(d_D, d_B);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_D, d_D, 64 * 32 * sizeof(float), cudaMemcpyDeviceToHost);

    errors = 0;
    for (int m = 0; m < 64; m++) {
        for (int n = 0; n < 32; n++) {
            float expected = 8.0f;
            float got = h_D[m * 32 + n];
            if (fabsf(got - expected) > 0.01f) {
                if (errors < 10) printf("MISMATCH D[%d][%d]: expected %.2f, got %.2f\n", m, n, expected, got);
                errors++;
            }
        }
    }
    if (errors == 0) printf("ALL CORRECT! D = 8.0 everywhere.\n");
    else printf("%d / %d mismatches\n", errors, 64*32);

    cudaFree(d_B);
    cudaFree(d_D);
    return 0;
}
