// Test: probe wgmma B matrix layout empirically
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr;\n"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}

constexpr uint32_t TF32_MASK = 0xFFFFE000u;

__device__ __forceinline__ uint64_t make_b_desc(void const* smem_ptr,
                                                  int lead_bytes, int stride_bytes) {
    uint32_t addr = smem_ptr_to_uint(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)(addr >> 4)) & 0x3FFF;
    desc |= ((uint64_t)((lead_bytes >> 4) & 0x3FFF)) << 16;
    desc |= ((uint64_t)((stride_bytes >> 4) & 0x3FFF)) << 32;
    return desc;
}

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

// Probe: set B[k][n] = 1000*k + n, A = all 1s
// D[m][n] = sum_k B_hw[k][n]
// From D values, figure out which stored elements map to each (k,n)
__global__ void __launch_bounds__(128)
probe_wgmma_kernel(float *D_out, int lead_bytes, int stride_bytes) {
    // B = 1024 bytes, stored flat. B_flat[i] for i in 0..255
    // We'll try different interpretations by varying the descriptor.
    __shared__ float B_smem[256];  // 8*32 = 256 floats

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int gid = lane / 4;
    int tid_in_grp = lane % 4;

    // Store B with unique values per byte offset
    // B_flat[i] = i (i = byte_offset / 4)
    for (int i = tid; i < 256; i += 128) {
        B_smem[i] = (float)i;
    }
    __syncthreads();

    uint32_t a_one = __float_as_uint(1.0f) & TF32_MASK;
    float d[16] = {};
    uint64_t desc_b = make_b_desc(&B_smem[0], lead_bytes, stride_bytes);

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_m64n32k8_rs(d, a_one, a_one, a_one, a_one, desc_b, 1);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    // Write D
    int row0 = warp_id * 16 + gid;
    int row1 = row0 + 8;
    for (int nb = 0; nb < 4; nb++) {
        int col0 = nb * 8 + 2 * tid_in_grp;
        int col1 = col0 + 1;
        D_out[row0 * 32 + col0] = d[4*nb + 0];
        D_out[row0 * 32 + col1] = d[4*nb + 1];
        D_out[row1 * 32 + col0] = d[4*nb + 2];
        D_out[row1 * 32 + col1] = d[4*nb + 3];
    }
}

// Same but with A having a single 1 at specific k position
__global__ void __launch_bounds__(128)
probe_single_k_kernel(float *D_out, int target_k, int lead_bytes, int stride_bytes) {
    __shared__ float B_smem[256];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int gid = lane / 4;
    int tid_in_grp = lane % 4;

    for (int i = tid; i < 256; i += 128) {
        B_smem[i] = (float)i;
    }
    __syncthreads();

    // A has 1.0 only at k=target_k. Which elements in A correspond to k=target_k?
    // In mma.sync m16n8k8 fragment: a0=(gid,2t), a1=(gid+8,2t), a2=(gid,2t+1), a3=(gid+8,2t+1)
    // For wgmma m64n32k8: same per-warp layout, warp w handles rows w*16..w*16+15
    // A[row][col] where row=warp*16+gid or +8, col=2*tid or 2*tid+1
    // We want A[any_row][target_k] = 1, rest = 0
    // target_k is the column (K dim). col = 2*tid or 2*tid+1.
    // For target_k = 2*t: tid_in_grp = t, a0 and a1 should be 1 (they cover col=2*t for both rows)
    // For target_k = 2*t+1: tid_in_grp = t, a2 and a3 should be 1

    uint32_t zero = 0;
    uint32_t one = __float_as_uint(1.0f) & TF32_MASK;

    uint32_t a0 = zero, a1 = zero, a2 = zero, a3 = zero;
    if (target_k / 2 == tid_in_grp) {
        if (target_k % 2 == 0) { a0 = one; a1 = one; }
        else                   { a2 = one; a3 = one; }
    }

    float d[16] = {};
    uint64_t desc_b = make_b_desc(&B_smem[0], lead_bytes, stride_bytes);

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_m64n32k8_rs(d, a0, a1, a2, a3, desc_b, 1);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    int row0 = warp_id * 16 + gid;
    int row1 = row0 + 8;
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
    float *d_D;
    cudaMalloc(&d_D, 64 * 32 * sizeof(float));
    float h_D[64 * 32];

    // Config: lead=128, stride=32 (row-major B[8][32])
    int lead = 128, stride = 32;

    printf("=== Single-k probes (lead=%d, stride=%d) ===\n", lead, stride);
    printf("B_smem[i] = i (flat index). D[0][n] = B_hw[target_k][n]\n\n");

    for (int k = 0; k < 8; k++) {
        probe_single_k_kernel<<<1, 128>>>(d_D, k, lead, stride);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error for k=%d: %s\n", k, cudaGetErrorString(err));
            break;
        }
        cudaMemcpy(h_D, d_D, 64 * 32 * sizeof(float), cudaMemcpyDeviceToHost);

        // D[0][n] gives B_hw[k][n] (since A is 1 only at k, all rows give same result)
        printf("k=%d: D[0] = [", k);
        for (int n = 0; n < 32; n++) {
            printf("%.0f", h_D[n]);
            if (n < 31) printf(",");
        }
        printf("]\n");
        // Interpret: B_hw[k][n] = h_D[n] means the hardware read element at flat index h_D[n]
        // This tells us which byte of B_smem the hardware reads for position (k,n)
    }

    // Also try column-major descriptor: lead=32 (8 floats * 4 bytes), stride=256 (8*32 bytes?)
    printf("\n=== Single-k probes (lead=32, stride=256) ===\n");
    for (int k = 0; k < 2; k++) {
        probe_single_k_kernel<<<1, 128>>>(d_D, k, 32, 256);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }
        cudaMemcpy(h_D, d_D, 64 * 32 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("k=%d: D[0] = [", k);
        for (int n = 0; n < 32; n++) {
            printf("%.0f", h_D[n]);
            if (n < 31) printf(",");
        }
        printf("]\n");
    }

    cudaFree(d_D);
    return 0;
}
