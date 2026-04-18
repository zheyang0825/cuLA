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

#include <ATen/cuda/CUDAContext.h>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/arch/arch.h>
#include <torch/extension.h>

#include "kda/sm90/bwd/kda_config.h"
#include "kda/sm90/prefill_kernel.hpp"

namespace sm90 {
void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream);
}

using OptionalTensor = std::optional<torch::Tensor>;

std::tuple<torch::Tensor, torch::Tensor>
kda_fwd_prefill(
    OptionalTensor output_,
    OptionalTensor output_state_,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    OptionalTensor input_state_,
    OptionalTensor alpha_,
    OptionalTensor beta_,
    torch::Tensor const& cu_seqlens,
    torch::Tensor workspace_buffer,
    float scale,
    bool safe_gate) {
    // Q, K, V: [packed_seq, H, D] (already packed by Python layer)
    auto packed_seq = q.size(0);
    auto num_heads = q.size(1);
    auto head_size = q.size(2);
    auto num_seqs = cu_seqlens.size(0) - 1;

    // KDA constraint: all head counts must be the same
    TORCH_CHECK(num_heads == k.size(1), "KDA requires num_q_heads == num_k_heads, got ", num_heads, " vs ", k.size(1));
    TORCH_CHECK(num_heads == v.size(1), "KDA requires num_q_heads == num_v_heads, got ", num_heads, " vs ", v.size(1));
    TORCH_CHECK(head_size == v.size(2), "KDA requires Q and V head dim to match, got ", head_size, " vs ", v.size(2));

    // Allocate output if not provided
    torch::Tensor output = output_.has_value() ? output_.value()
                                               : torch::empty(
                                                     {packed_seq, num_heads, head_size},
                                                     torch::TensorOptions().dtype(q.dtype()).device(q.device()));

    // Allocate output state if not provided
    torch::Tensor output_state = output_state_.has_value()
                                     ? output_state_.value()
                                     : torch::zeros(
                                           {num_seqs, num_heads, head_size, head_size},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    // Validate dtypes
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");

    // Validate contiguity
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output_state.is_contiguous(), "output_state must be contiguous");
    TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
    TORCH_CHECK(workspace_buffer.is_contiguous(), "workspace_buffer must be contiguous");

    // Extract optional pointers
    float const* alpha_ptr = nullptr;
    float const* input_state_ptr = nullptr;

    if (alpha_.has_value()) {
        auto& alpha = alpha_.value();
        TORCH_CHECK(alpha.dtype() == torch::kFloat32, "alpha must be float32");
        TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
        TORCH_CHECK(
            alpha.size(0) == packed_seq && alpha.size(1) == num_heads && alpha.size(2) == head_size,
            "alpha shape must be [packed_seq, num_heads, head_size]");
        alpha_ptr = alpha.data_ptr<float>();
    }
    if (beta_.has_value()) {
        auto& beta = beta_.value();
        TORCH_CHECK(
            beta.dtype() == torch::kFloat32 || beta.dtype() == torch::kBFloat16,
            "beta must be float32 or bfloat16, got ",
            beta.dtype());
        TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
        TORCH_CHECK(
            beta.size(0) == packed_seq && beta.size(1) == num_heads, "beta shape must be [packed_seq, num_heads]");
    }
    if (input_state_.has_value()) {
        auto& input_state = input_state_.value();
        TORCH_CHECK(input_state.dtype() == torch::kFloat32, "input_state must be float32");
        TORCH_CHECK(input_state.is_contiguous(), "input_state must be contiguous");
        input_state_ptr = input_state.data_ptr<float>();
    }

    // Auto-compute scale if 0
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    auto sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    using bf16 = cute::bfloat16_t;
    using Sm90 = cutlass::arch::Sm90;

    bool beta_is_bf16 = beta_.has_value() && beta_.value().dtype() == torch::kBFloat16;

    if (beta_is_bf16) {
        kda::sm90::launch_kda_fwd_prefill_kernel<Sm90, bf16, bf16, float, bf16>(
            stream,
            reinterpret_cast<bf16*>(output.data_ptr()),
            output_state.data_ptr<float>(),
            reinterpret_cast<bf16 const*>(q.data_ptr()),
            reinterpret_cast<bf16 const*>(k.data_ptr()),
            reinterpret_cast<bf16 const*>(v.data_ptr()),
            input_state_ptr,
            alpha_ptr,
            reinterpret_cast<bf16 const*>(beta_.value().data_ptr()),
            cu_seqlens.data_ptr<int32_t>(),
            workspace_buffer.data_ptr<uint8_t>(),
            static_cast<int32_t>(num_seqs),
            static_cast<int32_t>(num_heads),
            static_cast<int32_t>(head_size),
            static_cast<int64_t>(packed_seq),
            scale,
            safe_gate,
            static_cast<int32_t>(sm_count));
    } else {
        float const* beta_ptr = beta_.has_value() ? beta_.value().data_ptr<float>() : nullptr;
        kda::sm90::launch_kda_fwd_prefill_kernel<Sm90, bf16, bf16, float, float>(
            stream,
            reinterpret_cast<bf16*>(output.data_ptr()),
            output_state.data_ptr<float>(),
            reinterpret_cast<bf16 const*>(q.data_ptr()),
            reinterpret_cast<bf16 const*>(k.data_ptr()),
            reinterpret_cast<bf16 const*>(v.data_ptr()),
            input_state_ptr,
            alpha_ptr,
            beta_ptr,
            cu_seqlens.data_ptr<int32_t>(),
            workspace_buffer.data_ptr<uint8_t>(),
            static_cast<int32_t>(num_seqs),
            static_cast<int32_t>(num_heads),
            static_cast<int32_t>(head_size),
            static_cast<int64_t>(packed_seq),
            scale,
            safe_gate,
            static_cast<int32_t>(sm_count));
    }

    return {output, output_state};
}

void
ChunkKDABwdIntra(
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

    int num_chunks = chunk_indices.size(0);
    int NK = params.d / 32;    // BK = 32
    int NC = chunk_size / 16;  // BC = 16
    params.tile_scheduler_params = NaiveTileScheduler::Params{num_chunks, params.h, NK, NC};

    sm90::run_kda_bwd_intra_sm90(params, at::cuda::getCurrentCUDAStream());
}
