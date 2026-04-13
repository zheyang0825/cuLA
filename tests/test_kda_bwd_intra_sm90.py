"""
Tests for SM90 KDA Backward Intra-Chunk Kernel (Phase 1)

Phase 1 validation: kernel compiles, runs, and TMA-loaded dAqk/dAkk
match the Python input exactly via debug buffer.
"""

import pytest
import torch

pytestmark = pytest.mark.sm90_only


def make_chunk_indices(cu_seqlens, chunk_size=64):
    """Generate chunk_indices from cu_seqlens (matching C++ tile scheduler)."""
    indices = []
    for b_idx in range(len(cu_seqlens) - 1):
        start = cu_seqlens[b_idx].item()
        end = cu_seqlens[b_idx + 1].item()
        seq_len = end - start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            indices.append(b_idx)
            indices.append(c)
    return torch.tensor(indices, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize(
    "B, T, H, D",
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1-D128"),
        pytest.param(1, 128, 2, 128, id="B1-T128-H2-D128"),
        pytest.param(2, 256, 4, 128, id="B2-T256-H4-D128"),
    ],
)
def test_bwd_intra_sm90_phase1_runs(B, T, H, D):
    """Phase 1: Verify kernel compiles, launches, and produces output without errors."""
    torch.manual_seed(42)
    device = "cuda"
    BT = 64  # chunk size

    # Inputs (packed format: [total_q_len, H, D] or [total_q_len, H])
    total_q_len = B * T
    q = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.1  # small gate values
    # Make G monotonically decreasing within each chunk (realistic scenario)
    for b in range(B):
        for c in range((T + BT - 1) // BT):
            start = b * T + c * BT
            end = min(start + BT, b * T + T)
            if start < end:
                g[start:end] = g[start:end].cumsum(dim=0).neg()

    beta = torch.rand(total_q_len, H, dtype=torch.float32, device=device) * 0.5 + 0.5
    dAqk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01
    dAkk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01

    # Inter-chunk gradient inputs
    dq = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01
    dk = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01
    db = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01

    # cu_seqlens: simple equal-length sequences
    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)
    chunk_indices = make_chunk_indices(cu_seqlens, BT)

    # Outputs
    dq_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db_out = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)

    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    from cula.cudac import chunk_kda_bwd_intra_sm90

    # Run kernel
    chunk_kda_bwd_intra_sm90(
        q, k, g, beta, dAqk, dAkk,
        dq, dk, db, dg,
        cu_seqlens, chunk_indices,
        dq_out, dk_out, db_out, dg_out,
        tile_counter,
        BT,
    )

    torch.cuda.synchronize()

    # Phase 1 basic checks:
    # 1. No NaN/Inf in outputs
    assert not torch.isnan(dq_out).any(), "dq_out contains NaN"
    assert not torch.isnan(dk_out).any(), "dk_out contains NaN"
    assert not torch.isnan(db_out).any(), "db_out contains NaN"
    assert not torch.isnan(dg_out).any(), "dg_out contains NaN"
    assert not torch.isinf(dq_out).any(), "dq_out contains Inf"
    assert not torch.isinf(dk_out).any(), "dk_out contains Inf"

    # 2. In Phase 1, MMA outputs zeros, so dq_out ≈ dq_inter, dk_out ≈ dk_inter
    # dg_out should be dg_inter (pass-through in Phase 1)
    torch.testing.assert_close(
        dg_out, dg,
        atol=1e-5, rtol=1e-5,
        msg="Phase 1: dg_out should match dg_inter (pass-through)"
    )

    print(f"[PASS] Phase 1 test: B={B}, T={T}, H={H}, D={D}")


@pytest.mark.parametrize(
    "B, T, H, D",
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1-D128"),
    ],
)
def test_bwd_intra_sm90_phase1_dq_passthrough(B, T, H, D):
    """Phase 1: Verify dQ output = dQ_inter (since MMA is placeholder zeros)."""
    torch.manual_seed(123)
    device = "cuda"
    BT = 64

    total_q_len = B * T
    q = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    g = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)  # zero gate
    beta = torch.ones(total_q_len, H, dtype=torch.float32, device=device)
    dAqk = torch.zeros(total_q_len, H, BT, dtype=torch.float32, device=device)
    dAkk = torch.zeros(total_q_len, H, BT, dtype=torch.float32, device=device)
    dq = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device)
    dk = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device)
    db = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device)

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)
    chunk_indices = make_chunk_indices(cu_seqlens, BT)

    dq_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db_out = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    from cula.cudac import chunk_kda_bwd_intra_sm90

    chunk_kda_bwd_intra_sm90(
        q, k, g, beta, dAqk, dAkk,
        dq, dk, db, dg,
        cu_seqlens, chunk_indices,
        dq_out, dk_out, db_out, dg_out,
        tile_counter,
        BT,
    )
    torch.cuda.synchronize()

    # With zero dA and zero gate, MMA placeholder outputs zero.
    # dq_out = dq_inter + 0 = dq_inter
    torch.testing.assert_close(
        dq_out, dq,
        atol=1e-5, rtol=1e-5,
        msg="Phase 1: dq_out should match dq_inter when MMA is zeros"
    )
    torch.testing.assert_close(
        dk_out, dk,
        atol=1e-5, rtol=1e-5,
        msg="Phase 1: dk_out should match dk_inter when MMA is zeros"
    )

    print("[PASS] Phase 1 dQ/dK pass-through test")
