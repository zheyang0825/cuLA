// Empirical test of mma.sync.aligned.m16n8k8 fragment layout
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float &d0, float &d1, float &d2, float &d3,
    float a0, float a1, float a2, float a3,
    float b0, float b1,
    float c0, float c1, float c2, float c3)
{
    uint32_t ua0 = __float_as_uint(a0) & 0xFFFFE000u;
    uint32_t ua1 = __float_as_uint(a1) & 0xFFFFE000u;
    uint32_t ua2 = __float_as_uint(a2) & 0xFFFFE000u;
    uint32_t ua3 = __float_as_uint(a3) & 0xFFFFE000u;
    uint32_t ub0 = __float_as_uint(b0) & 0xFFFFE000u;
    uint32_t ub1 = __float_as_uint(b1) & 0xFFFFE000u;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(ua0), "r"(ua1), "r"(ua2), "r"(ua3),
          "r"(ub0), "r"(ub1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// Test kernel: single warp, compute D = A @ B where A = identity[16x8], B = identity[8x8]
// Then each thread dumps its d0-d3 values to global memory
__global__ void test_mma_layout(float *out_d, float *out_a, float *out_b) {
    int lane = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    int groupID = lane / 4;
    int threadID = lane % 4;

    // ==========================================================
    // Hypothesis: A fragment layout is a0=A[groupID, threadID*2]
    // Let's set A = identity[16][8] (lower-left part) and see what D we get
    // ==========================================================

    // Set all A elements to encode their row,col position
    // A[m][k] = (m+1)*100 + (k+1), so we can identify which A element appears in D
    // But we need to know the fragment layout to set this correctly...
    // Instead: just set a0=a1=a2=a3 = 1 for ALL threads, making A = all-ones[16x8]
    // And B = identity[8x8]
    // Then D = A @ B = A column sums... no, D = all-ones @ identity = all-ones
    // That's not helpful.

    // Better: set A = identity and B = identity
    // D = identity @ identity = identity
    // So D[m][n] = 1 if m==n (and m<8 since N=8), else 0
    // Then check which threads have non-zero d0/d1/d2/d3

    // Hypothesis 1: a0 = A[groupID, threadID*2], a2 = A[groupID+8, threadID*2]
    // Under this, for identity A:
    //   a0 = 1 if groupID == threadID*2 (never since groupID<=7 and threadID*2<=6, possible: g=0,t=0; g=2,t=1; g=4,t=2; g=6,t=3)
    //   But col is threadID*2, row is groupID. Identity: A[g][t*2] = 1 if g == t*2
    //   g=0,t=0: a0=A[0][0]=1 ✓
    //   g=2,t=1: a0=A[2][2]=1 ✓
    //   g=4,t=2: a0=A[4][4]=1 ✓
    //   g=6,t=3: a0=A[6][6]=1 ✓

    // Instead, let's use a unique encoding: A[m][k] = m*10 + k + 1
    // And B = identity[8x8]
    // Then D[m][n] = A[m][n] = m*10 + n + 1 (for n < 8)
    // By checking which d0,d1,d2,d3 values each thread gets, we can determine the layout.

    // Try hypothesis 1: a0 = A[groupID, threadID*2]
    float a0 = (float)(groupID * 10 + threadID * 2 + 1);
    float a1 = (float)(groupID * 10 + threadID * 2 + 1 + 1);
    float a2 = (float)((groupID + 8) * 10 + threadID * 2 + 1);
    float a3 = (float)((groupID + 8) * 10 + threadID * 2 + 1 + 1);

    // B = identity [8x8] in col-major fragment layout
    // b0 = B[threadID*2, groupID], b1 = B[threadID*2+1, groupID]
    float b0 = (threadID * 2 == groupID) ? 1.0f : 0.0f;
    float b1 = (threadID * 2 + 1 == groupID) ? 1.0f : 0.0f;

    // Store inputs for debugging
    out_a[lane * 4 + 0] = a0;
    out_a[lane * 4 + 1] = a1;
    out_a[lane * 4 + 2] = a2;
    out_a[lane * 4 + 3] = a3;
    out_b[lane * 2 + 0] = b0;
    out_b[lane * 2 + 1] = b1;

    float d0, d1, d2, d3;
    mma_m16n8k8_tf32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0.0f, 0.0f, 0.0f, 0.0f);

    out_d[lane * 4 + 0] = d0;
    out_d[lane * 4 + 1] = d1;
    out_d[lane * 4 + 2] = d2;
    out_d[lane * 4 + 3] = d3;
}

int main() {
    float *d_out_d, *d_out_a, *d_out_b;
    cudaMalloc(&d_out_d, 32 * 4 * sizeof(float));
    cudaMalloc(&d_out_a, 32 * 4 * sizeof(float));
    cudaMalloc(&d_out_b, 32 * 2 * sizeof(float));

    test_mma_layout<<<1, 32>>>(d_out_d, d_out_a, d_out_b);
    cudaDeviceSynchronize();

    float h_d[128], h_a[128], h_b[64];
    cudaMemcpy(h_d, d_out_d, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a, d_out_a, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_out_b, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== MMA m16n8k8 fragment layout test ===\n");
    printf("A[m][k] = m*10 + k + 1 (hypothesis: a0=A[gid, tid*2], a2=A[gid+8, tid*2])\n");
    printf("B = identity[8x8]\n");
    printf("Expected D = A (since D = A @ I)\n");
    printf("So d0 should equal A[row_d0, col_d0], telling us the D fragment layout.\n\n");

    printf("If hypothesis is correct:\n");
    printf("  d0 = D[gid, tid*2] = gid*10 + tid*2 + 1\n");
    printf("  d1 = D[gid, tid*2+1] = gid*10 + tid*2 + 2\n");
    printf("  d2 = D[gid+8, tid*2] = (gid+8)*10 + tid*2 + 1\n");
    printf("  d3 = D[gid+8, tid*2+1] = (gid+8)*10 + tid*2 + 2\n\n");

    for (int lane = 0; lane < 32; lane++) {
        int gid = lane / 4;
        int tid = lane % 4;
        printf("Lane %2d (gid=%d, tid=%d): d0=%6.0f d1=%6.0f d2=%6.0f d3=%6.0f | "
               "expected d0=%3d d1=%3d d2=%3d d3=%3d | "
               "match=%c%c%c%c\n",
               lane, gid, tid,
               h_d[lane*4+0], h_d[lane*4+1], h_d[lane*4+2], h_d[lane*4+3],
               gid*10+tid*2+1, gid*10+tid*2+2, (gid+8)*10+tid*2+1, (gid+8)*10+tid*2+2,
               (h_d[lane*4+0] == gid*10+tid*2+1) ? 'Y' : 'N',
               (h_d[lane*4+1] == gid*10+tid*2+2) ? 'Y' : 'N',
               (h_d[lane*4+2] == (gid+8)*10+tid*2+1) ? 'Y' : 'N',
               (h_d[lane*4+3] == (gid+8)*10+tid*2+2) ? 'Y' : 'N');
    }

    // Try to decode the actual layout from the output
    printf("\n=== Decoding actual D fragment layout ===\n");
    for (int lane = 0; lane < 32; lane++) {
        for (int di = 0; di < 4; di++) {
            float val = h_d[lane*4+di];
            if (val > 0) {
                int encoded = (int)(val + 0.5f);
                int row = (encoded - 1) / 10;
                int col = (encoded - 1) % 10;
                printf("  Lane %2d, d%d = %.0f → D[%d][%d]\n", lane, di, val, row, col);
            }
        }
    }

    cudaFree(d_out_d);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
    return 0;
}
