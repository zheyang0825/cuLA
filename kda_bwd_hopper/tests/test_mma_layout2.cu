// Test 2: verify the MMA layout by setting a0=a1=a2=a3=1, B=identity
// This avoids any ambiguity about A fragment layout
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

// Test: set a0=100, a1=200, a2=300, a3=400 (same for ALL threads)
// Set B = all-ones [8x8]
// Then D[m][n] = sum_k A[m][k] * B[k][n] = sum_k A[m][k]
// Each row of A is the same (since all threads set the same values and the
// fragment distributes across rows using groupID):
// A[gid, 2*tid] = 100 for all gid, tid  →  the EVEN columns of rows 0-7 are all 100
// A[gid, 2*tid+1] = 200                  →  the ODD columns of rows 0-7 are all 200
// A[gid+8, 2*tid] = 300                  →  the EVEN columns of rows 8-15 are all 300
// A[gid+8, 2*tid+1] = 400                →  the ODD columns of rows 8-15 are all 400
// Wait - that's wrong. Different threads with different tid set different columns.
// Each A element is set by exactly one thread.
//
// Actually: A[m][k] where m is determined by gid, k by tid.
// For row m=0 (gid=0): A[0][0] = a0 of lane(gid=0,tid=0) = 100
//                       A[0][1] = a1 of lane(gid=0,tid=0) = 200
//                       A[0][2] = a0 of lane(gid=0,tid=1) = 100
//                       A[0][3] = a1 of lane(gid=0,tid=1) = 200
//                       A[0][4] = a0 of lane(gid=0,tid=2) = 100
//                       ... etc
// So row 0 = [100, 200, 100, 200, 100, 200, 100, 200]
// Row 0 sum = 4*100 + 4*200 = 1200
//
// Row 8 (gid+8=8, from gid=0): A[8][0] = a2 of lane(gid=0,tid=0) = 300
//                                A[8][1] = a3 of lane(gid=0,tid=0) = 400
//                                ...
// Row 8 = [300, 400, 300, 400, 300, 400, 300, 400]
// Row 8 sum = 4*300 + 4*400 = 2800
//
// With B = all-ones:
// D[m][n] = row_sum(A[m]) for all n
// D[rows 0-7][any n] = 1200
// D[rows 8-15][any n] = 2800
//
// Now check which d register contains which value:
// If standard layout: d0 = D[gid, 2t] = 1200, d1 = D[gid, 2t+1] = 1200,
//                      d2 = D[gid+8, 2t] = 2800, d3 = D[gid+8, 2t+1] = 2800
// If swapped layout:  d0 = D[gid, 2t] = 1200, d1 = D[gid+8, 2t] = 2800,
//                      d2 = D[gid, 2t+1] = 1200, d3 = D[gid+8, 2t+1] = 2800
//
// So we can tell by whether d1 is 1200 (standard) or 2800 (swapped)!

__global__ void test_mma2(float *out) {
    int lane = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    float a0 = 100.0f, a1 = 200.0f, a2 = 300.0f, a3 = 400.0f;
    float b0 = 1.0f, b1 = 1.0f;  // All-ones B

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k8_tf32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    out[lane * 4 + 0] = d0;
    out[lane * 4 + 1] = d1;
    out[lane * 4 + 2] = d2;
    out[lane * 4 + 3] = d3;
}

// Test 2b: use distinct values for a0-a3 per thread to uniquely identify elements
// a0 = 1, a1 = 0, a2 = 0, a3 = 0 for all threads
// B = identity
// If standard A: A has 1s at (gid, 2*tid), zeros elsewhere
//   → each row 0-7 has exactly one 1 at col 2*tid (for the corresponding gid)
//   Actually each row m has one 1 at column 2*(lane%4) where lane is the one with gid=m
//   Wait, for row m: the 4 threads with gid=m are lanes 4m, 4m+1, 4m+2, 4m+3
//   Lane 4m (tid=0): a0→A[m][0]=1
//   Lane 4m+1 (tid=1): a0→A[m][2]=1
//   Lane 4m+2 (tid=2): a0→A[m][4]=1
//   Lane 4m+3 (tid=3): a0→A[m][6]=1
//   So row m = [1, 0, 1, 0, 1, 0, 1, 0]
//   Rows 8-15: a2=0 for all, so all zeros
//
// D = A @ I = A
// D[m][n] for m<8: = A[m][n]. So D[m][0]=1, D[m][1]=0, D[m][2]=1, etc.
// D[m][n] for m>=8: = 0
//
// Standard D: d0=D[g,2t]=A[g][2t]=1, d1=D[g,2t+1]=A[g][2t+1]=0,
//             d2=D[g+8,2t]=0, d3=D[g+8,2t+1]=0
//   So d = [1, 0, 0, 0]
//
// If A is swapped (a1↔a2): a0→A[g,2t]=1, a1→A[g+8,2t]=0, a2→A[g,2t+1]=0, a3→A[g+8,2t+1]=0
//   Same A as standard! (because a1=a2=0 anyway)
//   D = A @ I = A. Same result.
//
// If D is swapped: d0=D[g,2t]=1, d1=D[g+8,2t]=0, d2=D[g,2t+1]=0, d3=D[g+8,2t+1]=0
//   Same result! d = [1, 0, 0, 0] either way. Can't distinguish with a1=a2=0.
//
// So let's use a0=1, a1=0, a2=0, a3=0 AND a0=0, a1=10, a2=0, a3=0 in separate tests.

__global__ void test_mma_a_only(float *out, int which_a) {
    int lane = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    int gid = lane / 4;
    int tid = lane % 4;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (which_a == 0) a0 = (float)(gid * 10 + tid + 1);  // unique per lane
    if (which_a == 1) a1 = (float)(gid * 10 + tid + 1);
    if (which_a == 2) a2 = (float)(gid * 10 + tid + 1);
    if (which_a == 3) a3 = (float)(gid * 10 + tid + 1);

    // B = identity
    float b0 = (tid * 2 == gid) ? 1.0f : 0.0f;
    float b1 = (tid * 2 + 1 == gid) ? 1.0f : 0.0f;

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k8_tf32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    out[lane * 4 + 0] = d0;
    out[lane * 4 + 1] = d1;
    out[lane * 4 + 2] = d2;
    out[lane * 4 + 3] = d3;
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, 128 * sizeof(float));
    float h[128];

    // Test 1: constant a values, all-ones B
    printf("=== Test 1: a0=100, a1=200, a2=300, a3=400, B=all-ones ===\n");
    printf("Standard layout: d0=1200, d1=1200, d2=2800, d3=2800\n");
    printf("Swapped layout:  d0=1200, d1=2800, d2=1200, d3=2800\n\n");
    test_mma2<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Lane 0: d0=%.0f d1=%.0f d2=%.0f d3=%.0f\n", h[0], h[1], h[2], h[3]);
    printf("Lane 1: d0=%.0f d1=%.0f d2=%.0f d3=%.0f\n", h[4], h[5], h[6], h[7]);
    printf("Lane 4: d0=%.0f d1=%.0f d2=%.0f d3=%.0f\n", h[16], h[17], h[18], h[19]);
    if (h[1] == 1200.0f) printf("\nVERDICT: D fragment is STANDARD (d1→(gid, 2t+1))\n");
    else if (h[1] == 2800.0f) printf("\nVERDICT: D fragment is SWAPPED (d1→(gid+8, 2t))\n");
    else printf("\nVERDICT: UNEXPECTED d1=%.0f\n", h[1]);

    // Test 2: individual A registers with B=identity
    printf("\n=== Test 2: individual A registers, B=identity ===\n");
    const char* anames[] = {"a0", "a1", "a2", "a3"};
    for (int which = 0; which < 4; which++) {
        test_mma_a_only<<<1, 32>>>(d_out, which);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\nSetting only %s = unique_val, others = 0, B=identity:\n", anames[which]);
        for (int lane = 0; lane < 8; lane++) {
            printf("  Lane %d (g=%d,t=%d): d0=%.0f d1=%.0f d2=%.0f d3=%.0f\n",
                   lane, lane/4, lane%4,
                   h[lane*4], h[lane*4+1], h[lane*4+2], h[lane*4+3]);
        }
    }

    cudaFree(d_out);
    return 0;
}
