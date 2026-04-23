#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque C API for launching the SM90 KDA backward intra-chunk kernel.
 *
 * @param params  Pointer to a KDA_bwd_intra_params struct (defined in
 *                kda_bwd_common.h).  Kept as void* here so the header
 *                remains lightweight and does not pull in CUTLASS/CuTe.
 * @param stream  CUDA stream to launch on.
 */
void launch_c_kda_bwd_intra_sm90(void* params, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
