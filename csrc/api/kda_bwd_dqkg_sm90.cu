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

// Standalone pybind11 extension for the SM90 KDA Backward dqkg kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// Forward-declare the launcher and params from the kernel file.
struct KDABwdDqkgParams {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* v_new_ptr;
    const void* g_ptr;
    const void* beta_ptr;
    const void* A_ptr;
    const void* h_ptr;
    const void* do_ptr;
    const void* dh_ptr;
    const void* dv_ptr;
    void*       dq_ptr;
    void*       dk_ptr;
    void*       dv2_ptr;
    void*       dg_ptr;
    void*       db_ptr;
    void*       dA_ptr;
    int B, T, H;
    float scale;
};

void launch_chunk_kda_bwd_dqkg(const KDABwdDqkgParams& p, cudaStream_t stream);

// ---- Python binding ----
// Returns (dq, dk, dv2, dg, db, dA).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ChunkKDABwdDqkgSm90(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor v_new,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor A,
    at::Tensor h,
    at::Tensor do_,
    at::Tensor dh,
    at::Tensor dv,
    float scale)
{
    // ---- Dtype checks ----
    TORCH_CHECK(q.is_cuda(),     "q must be on CUDA");
    TORCH_CHECK(q.dtype()     == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype()     == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype()     == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(v_new.dtype() == torch::kBFloat16, "v_new must be bfloat16");
    TORCH_CHECK(g.dtype()     == torch::kFloat32,  "g must be float32");
    TORCH_CHECK(beta.dtype()  == torch::kFloat32,  "beta must be float32");
    TORCH_CHECK(A.dtype()     == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(h.dtype()     == torch::kBFloat16, "h must be bfloat16");
    TORCH_CHECK(do_.dtype()   == torch::kBFloat16, "do_ must be bfloat16");
    TORCH_CHECK(dh.dtype()    == torch::kBFloat16, "dh must be bfloat16");
    TORCH_CHECK(dv.dtype()    == torch::kBFloat16, "dv must be bfloat16");

    // ---- Contiguity checks ----
    TORCH_CHECK(q.is_contiguous(),     "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(),     "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(),     "v must be contiguous");
    TORCH_CHECK(v_new.is_contiguous(), "v_new must be contiguous");
    TORCH_CHECK(g.is_contiguous(),     "g must be contiguous");
    TORCH_CHECK(beta.is_contiguous(),  "beta must be contiguous");
    TORCH_CHECK(A.is_contiguous(),     "A must be contiguous");
    TORCH_CHECK(h.is_contiguous(),     "h must be contiguous");
    TORCH_CHECK(do_.is_contiguous(),   "do_ must be contiguous");
    TORCH_CHECK(dh.is_contiguous(),    "dh must be contiguous");
    TORCH_CHECK(dv.is_contiguous(),    "dv must be contiguous");

    // ---- Shape extraction ----
    // q: [B*T, H, K] (packed)
    TORCH_CHECK(q.dim() == 3, "q must be 3-D [B*T, H, K]");
    int BT_total = q.size(0);
    int H        = q.size(1);
    int K        = q.size(2);
    TORCH_CHECK(K == 128, "K must be 128");
    TORCH_CHECK(v.dim() == 3, "v must be 3-D [B*T, H, V]");
    int V = v.size(2);
    TORCH_CHECK(V == 128, "V must be 128");

    // beta: [B*T, H]
    TORCH_CHECK(beta.dim() == 2 && beta.size(0) == BT_total && beta.size(1) == H,
                "beta must be [B*T, H]");
    // A: [B*T, H, BT_chunk] where BT_chunk == 64
    TORCH_CHECK(A.dim() == 3 && A.size(0) == BT_total && A.size(1) == H && A.size(2) == 64,
                "A must be [B*T, H, 64]");

    // h/dh: [NT*B, H, K, V]
    TORCH_CHECK(h.dim()  == 4, "h must be 4-D [NT*B, H, K, V]");
    TORCH_CHECK(dh.dim() == 4, "dh must be 4-D [NT*B, H, K, V]");

    // We need B and T. The user must pass B via the shape; infer from h.
    // h.size(0) = NT * B where NT = ceil(T/64).
    // For simplicity, require q to be [B*T, H, K] and A.size(0) == B*T.
    // We accept B=1 for now and infer T = BT_total / B... we need B explicitly.
    // Pass B as an argument derived from the caller.
    // For now: accept B=1 (common test case). Use a separate overload if needed.
    // Actually: infer B from h.size(0) and T. NT = h.size(0)/B, T = NT*64.
    // Too circular. Let the caller pass T explicitly, and derive B = BT_total/T.
    TORCH_CHECK(false, "Use ChunkKDABwdDqkgSm90WithBT instead");

    // Unreachable
    at::Tensor dummy;
    return {dummy, dummy, dummy, dummy, dummy, dummy};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ChunkKDABwdDqkgSm90Run(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor v_new,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor A,
    at::Tensor h,
    at::Tensor do_,
    at::Tensor dh,
    at::Tensor dv,
    float scale,
    int B,
    int T)
{
    // ---- Dtype checks ----
    TORCH_CHECK(q.is_cuda(),     "q must be on CUDA");
    TORCH_CHECK(q.dtype()     == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype()     == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype()     == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(v_new.dtype() == torch::kBFloat16, "v_new must be bfloat16");
    TORCH_CHECK(g.dtype()     == torch::kFloat32,  "g must be float32");
    TORCH_CHECK(beta.dtype()  == torch::kFloat32,  "beta must be float32");
    TORCH_CHECK(A.dtype()     == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(h.dtype()     == torch::kBFloat16, "h must be bfloat16");
    TORCH_CHECK(do_.dtype()   == torch::kBFloat16, "do_ must be bfloat16");
    TORCH_CHECK(dh.dtype()    == torch::kBFloat16, "dh must be bfloat16");
    TORCH_CHECK(dv.dtype()    == torch::kBFloat16, "dv must be bfloat16");

    // ---- Contiguity ----
    TORCH_CHECK(q.is_contiguous(),     "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(),     "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(),     "v must be contiguous");
    TORCH_CHECK(v_new.is_contiguous(), "v_new must be contiguous");
    TORCH_CHECK(g.is_contiguous(),     "g must be contiguous");
    TORCH_CHECK(beta.is_contiguous(),  "beta must be contiguous");
    TORCH_CHECK(A.is_contiguous(),     "A must be contiguous");
    TORCH_CHECK(h.is_contiguous(),     "h must be contiguous");
    TORCH_CHECK(do_.is_contiguous(),   "do_ must be contiguous");
    TORCH_CHECK(dh.is_contiguous(),    "dh must be contiguous");
    TORCH_CHECK(dv.is_contiguous(),    "dv must be contiguous");

    // ---- Shape ----
    int H = q.size(1);
    int K = q.size(2);
    int V = v.size(2);
    TORCH_CHECK(K == 128, "K must be 128 for Phase 1 kernel");
    TORCH_CHECK(V == 128, "V must be 128 for Phase 1 kernel");
    TORCH_CHECK(q.size(0) == (int64_t)B * T, "q.size(0) must equal B*T");

    // ---- Allocate outputs ----
    at::Tensor dq  = torch::empty_like(q,    torch::kFloat);
    at::Tensor dk  = torch::empty_like(k,    torch::kFloat);
    at::Tensor dv2 = torch::empty_like(v);   // bf16, same as v
    at::Tensor dg  = torch::empty_like(g,    torch::kFloat);
    at::Tensor db  = torch::empty_like(beta, torch::kFloat);
    at::Tensor dA  = torch::empty_like(A,    torch::kFloat);

    // ---- Launch ----
    KDABwdDqkgParams params;
    params.q_ptr     = q.data_ptr();
    params.k_ptr     = k.data_ptr();
    params.v_ptr     = v.data_ptr();
    params.v_new_ptr = v_new.data_ptr();
    params.g_ptr     = g.data_ptr();
    params.beta_ptr  = beta.data_ptr();
    params.A_ptr     = A.data_ptr();
    params.h_ptr     = h.data_ptr();
    params.do_ptr    = do_.data_ptr();
    params.dh_ptr    = dh.data_ptr();
    params.dv_ptr    = dv.data_ptr();
    params.dq_ptr    = dq.data_ptr();
    params.dk_ptr    = dk.data_ptr();
    params.dv2_ptr   = dv2.data_ptr();
    params.dg_ptr    = dg.data_ptr();
    params.db_ptr    = db.data_ptr();
    params.dA_ptr    = dA.data_ptr();
    params.B         = B;
    params.T         = T;
    params.H         = H;
    params.scale     = scale;

    auto stream = at::cuda::getCurrentCUDAStream();
    c10::cuda::CUDAGuard device_guard(q.device());

    launch_chunk_kda_bwd_dqkg(params, stream);

    return {dq, dk, dv2, dg, db, dA};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuLA SM90 KDA Backward dqkg standalone";
    m.def(
        "chunk_kda_bwd_dqkg_sm90",
        &ChunkKDABwdDqkgSm90Run,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("v_new"),
        py::arg("g"),
        py::arg("beta"),
        py::arg("A"),
        py::arg("h"),
        py::arg("do_"),
        py::arg("dh"),
        py::arg("dv"),
        py::arg("scale"),
        py::arg("B"),
        py::arg("T"),
        "Run SM90 KDA backward dqkg fused kernel. Returns (dq, dk, dv2, dg, db, dA).");
}
