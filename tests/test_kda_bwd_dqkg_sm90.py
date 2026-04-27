"""
Tests for SM90 KDA Backward dqkg Kernel (Phase 1).

Compares CUDA output against the FLA Triton reference implementation
chunk_kda_bwd_wy_dqkg_fused.
"""

import math
import pytest
import torch

pytestmark = pytest.mark.sm90_only


def _triton_ref(q, k, v, v_new, g, beta, A, h, do_, dh, dv, scale, B, T):
    """Run FLA Triton reference kernel."""
    import sys
    sys.path.insert(0, "third_party/flash-linear-attention")
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused

    # Triton expects [B, T, H, K/V] layout — reshape from [B*T, H, K/V]
    H_ = q.shape[1]
    K_ = q.shape[2]
    V_ = v.shape[2]

    q4    = q   .reshape(B, T, H_, K_)
    k4    = k   .reshape(B, T, H_, K_)
    v4    = v   .reshape(B, T, H_, V_)
    vnew4 = v_new.reshape(B, T, H_, V_)
    g4    = g   .reshape(B, T, H_, K_)
    b4    = beta.reshape(B, T, H_)
    A4    = A   .reshape(B, T, H_, 64)
    do4   = do_ .reshape(B, T, H_, V_)
    dv4   = dv  .reshape(B, T, H_, V_)

    # h / dh: Triton expects [NT*B, H, K, V]
    # h is already [NT*B, H, K, V] from kernel convention

    return chunk_kda_bwd_wy_dqkg_fused(
        q=q4, k=k4, v=v4, v_new=vnew4, g=g4, beta=b4,
        A=A4, h=h, do=do4, dh=dh, dv=dv4,
        scale=scale, chunk_size=64,
        transpose_state_layout=False)


def _make_inputs(B, T, H, K, V, seed=42):
    torch.manual_seed(seed)
    dev = "cuda"
    bf = torch.bfloat16
    fp = torch.float32

    q     = torch.randn(B * T, H, K, dtype=bf, device=dev)
    k     = torch.randn(B * T, H, K, dtype=bf, device=dev)
    v     = torch.randn(B * T, H, V, dtype=bf, device=dev)
    v_new = torch.randn(B * T, H, V, dtype=bf, device=dev)
    # g in log2 domain — use small negatives for stability
    g     = -torch.rand( B * T, H, K, dtype=fp, device=dev) * 0.5
    beta  = torch.rand( B * T, H,    dtype=fp, device=dev) * 0.5 + 0.1
    # A: lower-triangular structure typical for KDA
    A_raw = torch.randn(B * T, H, 64, dtype=bf, device=dev) * 0.1

    NT = (T + 63) // 64
    h   = torch.randn(NT * B, H, K, V, dtype=bf, device=dev) * 0.1
    dh  = torch.randn(NT * B, H, K, V, dtype=bf, device=dev) * 0.1

    do_ = torch.randn(B * T, H, V, dtype=bf, device=dev)
    dv  = torch.randn(B * T, H, V, dtype=bf, device=dev)

    return q, k, v, v_new, g, beta, A_raw, h, do_, dh, dv


@pytest.mark.parametrize(
    "B, T, H",
    [
        pytest.param(1,  64, 1, id="B1-T64-H1"),
        pytest.param(1, 128, 2, id="B1-T128-H2"),
        pytest.param(2, 256, 4, id="B2-T256-H4"),
    ],
)
def test_dqkg_sm90_vs_triton(B, T, H):
    """CUDA output must match Triton reference within tolerance."""
    K, V = 128, 128
    scale = 1.0 / math.sqrt(K)

    q, k, v, v_new, g, beta, A, h, do_, dh, dv = _make_inputs(B, T, H, K, V)

    # ---- CUDA kernel ----
    from cula._kda_bwd_dqkg_sm90 import chunk_kda_bwd_dqkg_sm90
    cuda_dq, cuda_dk, cuda_dv2, cuda_dg, cuda_db, cuda_dA = chunk_kda_bwd_dqkg_sm90(
        q, k, v, v_new, g, beta, A, h, do_, dh, dv, scale, B, T)

    # ---- Triton reference ----
    ref_dq, ref_dk, ref_dv2, ref_db, ref_dg, ref_dA = _triton_ref(
        q, k, v, v_new, g, beta, A, h, do_, dh, dv, scale, B, T)

    # Triton outputs are [B, T, H, K/V]; reshape to [B*T, H, K/V] for comparison
    def flat(x):
        if x.dim() == 4:
            b, t, hh, d = x.shape
            return x.reshape(b * t, hh, d)
        elif x.dim() == 3:
            b, t, hh = x.shape
            return x.reshape(b * t, hh)
        return x

    ref_dq  = flat(ref_dq) .float()
    ref_dk  = flat(ref_dk) .float()
    ref_dv2 = flat(ref_dv2).float()
    ref_dg  = flat(ref_dg) .float()
    ref_db  = flat(ref_db) .float()
    ref_dA  = flat(ref_dA) .float()

    cuda_dq  = cuda_dq .float()
    cuda_dk  = cuda_dk .float()
    cuda_dv2 = cuda_dv2.float()
    cuda_dg  = cuda_dg .float()
    cuda_db  = cuda_db .float()
    cuda_dA  = cuda_dA .float()

    # dq/dk: direct bf16→fp32 products, tight tolerance.
    tol_qk = dict(rtol=1e-2, atol=1e-3)
    assert torch.allclose(cuda_dq,  ref_dq,  **tol_qk), \
        f"dq mismatch: max={( cuda_dq  - ref_dq ).abs().max():.4e}"
    assert torch.allclose(cuda_dk,  ref_dk,  **tol_qk), \
        f"dk mismatch: max={( cuda_dk  - ref_dk ).abs().max():.4e}"
    assert torch.allclose(cuda_dv2, ref_dv2, **tol_qk), \
        f"dv2 mismatch: max={( cuda_dv2 - ref_dv2).abs().max():.4e}"

    # dg/db/dA: multi-step accumulation from bf16 → fp32; scalar fp32 loops
    # vs Triton tensor-core bf16 arithmetic cause O(0.1%) rounding differences.
    tol_rest = dict(rtol=2e-2, atol=5e-2)
    assert torch.allclose(cuda_dg,  ref_dg,  **tol_rest), \
        f"dg mismatch: max={( cuda_dg  - ref_dg ).abs().max():.4e}"
    assert torch.allclose(cuda_db,  ref_db,  **tol_rest), \
        f"db mismatch: max={( cuda_db  - ref_db ).abs().max():.4e}"
    assert torch.allclose(cuda_dA,  ref_dA,  **tol_rest), \
        f"dA mismatch: max={( cuda_dA  - ref_dA ).abs().max():.4e}"
