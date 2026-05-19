#include <torch/python.h>
#include <tuple>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kda_bwd/kda_config.h"

void ChunkKDABwdIntra(
    at::Tensor q,
    at::Tensor k,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor dAqk,
    at::Tensor dAkk,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor db,
    at::Tensor dg,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor dq_out,
    at::Tensor dk_out,
    at::Tensor db_out,
    at::Tensor dg_out,
    int chunk_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_kda_bwd_intra_cuda", &ChunkKDABwdIntra);
}
