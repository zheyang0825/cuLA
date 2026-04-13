// SM90 KDA Backward Intra-Chunk — PyTorch C++ API Entry Point

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kda/sm90/bwd/kda_bwd_intra_sm90.cuh"

void
ChunkKDABwdIntraSm90(
    at::Tensor q,              // [total_q_len, h, d] bf16
    at::Tensor k,              // [total_q_len, h, d] bf16
    at::Tensor g,              // [total_q_len, h, d] fp32
    at::Tensor beta,           // [total_q_len, h] fp32
    at::Tensor dAqk,           // [total_q_len, h, BT] fp32
    at::Tensor dAkk,           // [total_q_len, h, BT] fp32
    at::Tensor dq,             // [total_q_len, h, d] fp32 (inter-chunk dQ input)
    at::Tensor dk,             // [total_q_len, h, d] fp32 (inter-chunk dK input)
    at::Tensor db,             // [total_q_len, h] fp32 (inter-chunk dB input)
    at::Tensor dg,             // [total_q_len, h, d] fp32 (inter-chunk dG input)
    at::Tensor cu_seqlens,     // [b+1] int32
    at::Tensor chunk_indices,  // [num_chunks * 2] int32
    at::Tensor dq_out,         // [total_q_len, h, d] fp32 (output)
    at::Tensor dk_out,         // [total_q_len, h, d] fp32 (output)
    at::Tensor db_out,         // [total_q_len, h] fp32 (output)
    at::Tensor dg_out,         // [total_q_len, h, d] fp32 (output)
    at::Tensor tile_counter,   // [1] int32
    int chunk_size) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be float32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(dAqk.dtype() == torch::kFloat32, "dAqk must be float32");
    TORCH_CHECK(dAkk.dtype() == torch::kFloat32, "dAkk must be float32");
    TORCH_CHECK(dq.dtype() == torch::kFloat32, "dq must be float32");
    TORCH_CHECK(dk.dtype() == torch::kFloat32, "dk must be float32");
    TORCH_CHECK(dg.dtype() == torch::kFloat32, "dg must be float32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32, "chunk_indices must be int32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(dAqk.is_contiguous(), "dAqk must be contiguous");
    TORCH_CHECK(dAkk.is_contiguous(), "dAkk must be contiguous");
    TORCH_CHECK(dq.is_contiguous(), "dq must be contiguous");
    TORCH_CHECK(dk.is_contiguous(), "dk must be contiguous");
    TORCH_CHECK(dg.is_contiguous(), "dg must be contiguous");
    TORCH_CHECK(dq_out.is_contiguous(), "dq_out must be contiguous");
    TORCH_CHECK(dk_out.is_contiguous(), "dk_out must be contiguous");
    TORCH_CHECK(dg_out.is_contiguous(), "dg_out must be contiguous");

    int total_q_len = q.size(0);
    int h = q.size(1);
    int d = q.size(2);
    int b = cu_seqlens.size(0) - 1;
    int num_chunks = chunk_indices.size(0) / 2;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    // Zero tile counter
    tile_counter.zero_();

    KDA_bwd_intra_params params;
    params.total_q_len = total_q_len;
    params.b = b;
    params.h = h;
    params.d = d;
    params.d_v = d;
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
    params.dq_out_ptr = dq_out.data_ptr();
    params.dk_out_ptr = dk_out.data_ptr();
    params.db_out_ptr = db_out.data_ptr();
    params.dg_out_ptr = dg_out.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();

    params.tile_scheduler_params.num_blocks = num_chunks;
    params.tile_scheduler_params.num_heads = h;
    params.tile_scheduler_params.num_sm = sm_count;
    params.tile_scheduler_params.tile_counter = tile_counter.data_ptr<int>();
    params.num_sm = sm_count;

    sm90_bwd::run_kda_bwd_intra_sm90(params, stream);
}
