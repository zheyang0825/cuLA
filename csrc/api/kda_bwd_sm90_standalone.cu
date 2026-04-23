// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Standalone extension for SM90 KDA Backward Intra-Chunk kernel.
// Compiles independently from the rest of cula.cudac to iterate faster.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kda/sm90/bwd/kda_bwd_common.h"
#include "kda/sm90/bwd/kda_bwd_intra_sm90_api.h"

void
ChunkKDABwdIntraSm90(
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
    at::Tensor tile_counter,
    int chunk_size,
    std::optional<at::Tensor> debug_kg,
    std::optional<at::Tensor> debug_qg,
    std::optional<at::Tensor> debug_kbg)
{
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

    params.debug_kg_ptr = debug_kg.has_value() ? debug_kg->data_ptr() : nullptr;
    params.debug_qg_ptr = debug_qg.has_value() ? debug_qg->data_ptr() : nullptr;
    params.debug_kbg_ptr = debug_kbg.has_value() ? debug_kbg->data_ptr() : nullptr;

    params.tile_scheduler_params.num_blocks = num_chunks;
    params.tile_scheduler_params.num_heads = h;
    params.tile_scheduler_params.num_sm = sm_count;
    params.tile_scheduler_params.tile_counter = tile_counter.data_ptr<int>();
    params.num_sm = sm_count;

    launch_c_kda_bwd_intra_sm90(static_cast<void*>(&params), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuLA SM90 KDA Backward Intra standalone";
    m.def(
        "chunk_kda_bwd_intra_sm90",
        &ChunkKDABwdIntraSm90,
        py::arg("q"),
        py::arg("k"),
        py::arg("g"),
        py::arg("beta"),
        py::arg("dAqk"),
        py::arg("dAkk"),
        py::arg("dq"),
        py::arg("dk"),
        py::arg("db"),
        py::arg("dg"),
        py::arg("cu_seqlens"),
        py::arg("chunk_indices"),
        py::arg("dq_out"),
        py::arg("dk_out"),
        py::arg("db_out"),
        py::arg("dg_out"),
        py::arg("tile_counter"),
        py::arg("chunk_size"),
        py::arg("debug_kg") = py::none(),
        py::arg("debug_qg") = py::none(),
        py::arg("debug_kbg") = py::none());
}
