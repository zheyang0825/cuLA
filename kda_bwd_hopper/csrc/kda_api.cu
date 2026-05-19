#include <torch/python.h>
#include "kda_bwd/kda_config.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace sm90 {
void run_kda_bwd_intra_sm90(KDA_bwd_intra_params &params, cudaStream_t stream);
}

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
    int chunk_size) {

    constexpr int BK = 32;

    KDA_bwd_intra_params params;
    params.total_q_len = q.size(0) * q.size(1);
    params.b = cu_seqlens.size(0) - 1;
    params.h = q.size(2);
    params.d = q.size(3);
    params.chunk_size = chunk_size;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.dAqk_ptr = dAqk.data_ptr();
    params.dAkk_ptr = dAkk.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.db_ptr = db.data_ptr();
    params.dg_ptr = dg.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.dq_out_ptr = dq_out.data_ptr();
    params.dk_out_ptr = dk_out.data_ptr();
    params.db_out_ptr = db_out.data_ptr();
    params.dg_out_ptr = dg_out.data_ptr();
    params.num_chunks = chunk_indices.size(0);

    int NK = (params.d + BK - 1) / BK;
    params.num_k_tiles = NK;

    // Allocate per-K-tile db buffer for deterministic reduction
    auto db2 = torch::zeros({NK, db.size(0), db.size(1), db.size(2)},
                            torch::TensorOptions().dtype(torch::kFloat32).device(db.device()));
    params.db2_ptr = db2.data_ptr();

    sm90::run_kda_bwd_intra_sm90(params, at::cuda::getCurrentCUDAStream());

    // Reduce db2 across K-tiles and add inter-chunk db
    db_out.copy_(db2.sum(0).add_(db));
}
