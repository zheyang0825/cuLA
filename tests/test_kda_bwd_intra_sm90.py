"""
Tests for SM90 KDA Backward Intra-Chunk Kernel

Phase 1: kernel compiles, runs, pass-through outputs match inter-chunk inputs.
Phase 2: Prep B operands (KG/QG/KBG) match Python reference via debug buffers.
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
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
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

    # 2. Non-zero output (kernel actually computed something)
    assert dq_out.abs().sum() > 0, "dq_out is all zeros"
    assert dk_out.abs().sum() > 0, "dk_out is all zeros"

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
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        BT,
    )
    torch.cuda.synchronize()

    # With zero dA and zero gate, MMA placeholder outputs zero.
    # dq_out = dq_inter + 0 = dq_inter
    torch.testing.assert_close(dq_out, dq, atol=1e-5, rtol=1e-5, msg="Phase 1: dq_out should match dq_inter when MMA is zeros")
    torch.testing.assert_close(dk_out, dk, atol=1e-5, rtol=1e-5, msg="Phase 1: dk_out should match dk_inter when MMA is zeros")

    print("[PASS] Phase 1 dQ/dK pass-through test")


# =====================================================================
# Phase 2: B Operand (KG / QG / KBG) verification
# =====================================================================


def compute_b_operands_ref(q, k, g, beta, BT=64, SUB_T=16):
    """Compute KG, QG, KBG reference in Python.

    Args:
        q: [T, D] bf16 (one chunk, one head)
        k: [T, D] bf16
        g: [T, D] fp32
        beta: [T] fp32
    Returns:
        kg, qg, kbg: [T, D] fp32
    """
    T, D = g.shape
    k_f = k.float()
    q_f = q.float()
    kg = torch.zeros(T, D, dtype=torch.float32, device=g.device)
    qg = torch.zeros(T, D, dtype=torch.float32, device=g.device)
    kbg = torch.zeros(T, D, dtype=torch.float32, device=g.device)

    num_sub = T // SUB_T
    for s in range(num_sub):
        row_start = s * SUB_T
        row_end = row_start + SUB_T
        g_norm = g[row_start, :]  # [D]

        # KG: K * exp2(G_norm - G)
        kg[row_start:row_end] = k_f[row_start:row_end] * torch.exp2(g_norm - g[row_start:row_end])
        # QG: Q * exp2(G - G_norm)
        qg[row_start:row_end] = q_f[row_start:row_end] * torch.exp2(g[row_start:row_end] - g_norm)
        # KBG: K * beta * exp2(G - G_norm)
        kbg[row_start:row_end] = (
            k_f[row_start:row_end] * beta[row_start:row_end, None] * torch.exp2(g[row_start:row_end] - g_norm)
        )

    return kg, qg, kbg


@pytest.mark.parametrize(
    "B, T, H, D",
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1-D128"),
        pytest.param(1, 64, 2, 128, id="B1-T64-H2-D128"),
        pytest.param(2, 128, 2, 128, id="B2-T128-H2-D128"),
    ],
)
def test_bwd_intra_sm90_phase2_b_operands(B, T, H, D):
    """Phase 2: Verify KG/QG/KBG from Prep match Python reference."""
    torch.manual_seed(42)
    device = "cuda"
    BT = 64

    total_q_len = B * T
    q = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.1
    for b in range(B):
        for c in range((T + BT - 1) // BT):
            start = b * T + c * BT
            end = min(start + BT, b * T + T)
            if start < end:
                g[start:end] = g[start:end].cumsum(dim=0).neg()

    beta = torch.rand(total_q_len, H, dtype=torch.float32, device=device) * 0.5 + 0.5
    dAqk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01
    dAkk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01
    dq = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01
    dk = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01
    db = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)
    chunk_indices = make_chunk_indices(cu_seqlens, BT)

    # Outputs (required but not checked here)
    dq_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db_out = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    # Debug output tensors
    debug_kg = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    debug_qg = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    debug_kbg = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)

    from cula.cudac import chunk_kda_bwd_intra_sm90

    chunk_kda_bwd_intra_sm90(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        BT,
        debug_kg,
        debug_qg,
        debug_kbg,
    )
    torch.cuda.synchronize()

    # Verify per-chunk, per-head
    num_chunks = T // BT
    for b_idx in range(B):
        for c in range(num_chunks):
            for h in range(H):
                start = b_idx * T + c * BT
                end = start + BT

                # Extract single chunk, single head
                q_ch = q[start:end, h, :]  # [64, D] bf16
                k_ch = k[start:end, h, :]  # [64, D] bf16
                g_ch = g[start:end, h, :]  # [64, D] fp32
                beta_ch = beta[start:end, h]  # [64] fp32

                kg_ref, qg_ref, kbg_ref = compute_b_operands_ref(q_ch, k_ch, g_ch, beta_ch)

                kg_cuda = debug_kg[start:end, h, :]
                qg_cuda = debug_qg[start:end, h, :]
                kbg_cuda = debug_kbg[start:end, h, :]

                torch.testing.assert_close(
                    kg_cuda,
                    kg_ref,
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"KG mismatch: b={b_idx}, c={c}, h={h}",
                )
                torch.testing.assert_close(
                    qg_cuda,
                    qg_ref,
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"QG mismatch: b={b_idx}, c={c}, h={h}",
                )
                torch.testing.assert_close(
                    kbg_cuda,
                    kbg_ref,
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"KBG mismatch: b={b_idx}, c={c}, h={h}",
                )

    print(f"[PASS] Phase 2 B operand test: B={B}, T={T}, H={H}, D={D}")


# =====================================================================
# Phase 3: MMA 4-pass output verification
# =====================================================================


def truncate_to_tf32(x):
    """Truncate fp32 to TF32 (keep upper 10 bits of mantissa, zero lower 13)."""
    import numpy as np

    x_np = x.detach().cpu().float().numpy()
    x_bytes = x_np.view(np.uint32)
    x_bytes = x_bytes & np.uint32(0xFFFFE000)
    return torch.from_numpy(x_bytes.view(np.float32)).to(x.device)


def compute_dq_dk_intra_ref(q, k, g, beta, dAqk, dAkk, BT=64, SUB_T=16):
    """Compute dQ_intra and dK_intra (=dK_lower + dK_upper) reference in Python.

    Simulates the 4-pass mma.sync sub-chunk loop with TF32 truncation.

    Args:
        q: [T, D] bf16 (one chunk, one head)
        k: [T, D] bf16
        g: [T, D] fp32
        beta: [T] fp32
        dAqk: [T, T] fp32
        dAkk: [T, T] fp32
    Returns:
        dQ_intra, dK_intra: [T, D] fp32
    """
    T, D = q.shape
    q_f = q.float()
    k_f = k.float()
    num_sub = T // SUB_T  # 4

    dQ = torch.zeros(T, D, dtype=torch.float32, device=q.device)
    dK = torch.zeros(T, D, dtype=torch.float32, device=q.device)

    K_TILE = 32
    for d_start in range(0, D, K_TILE):
        d_end = d_start + K_TILE
        g_slice = g[:, d_start:d_end]
        k_slice = k_f[:, d_start:d_end]
        q_slice = q_f[:, d_start:d_end]

        # Pass 1: dQ = tril(dAqk, 0) × KG, with per-subchunk scaling
        for is_ in range(num_sub):
            for js in range(is_ + 1):
                g_norm = g_slice[js * SUB_T, :]  # [K_TILE]
                KG_js = k_slice[js * SUB_T : (js + 1) * SUB_T, :] * torch.exp2(
                    g_norm - g_slice[js * SUB_T : (js + 1) * SUB_T, :]
                )

                A_sub = dAqk[is_ * SUB_T : (is_ + 1) * SUB_T, js * SUB_T : (js + 1) * SUB_T].clone()
                if is_ == js:
                    mask = torch.triu(torch.ones(SUB_T, SUB_T, device=q.device), diagonal=1).bool()
                    A_sub[mask] = 0.0

                A_tf32 = truncate_to_tf32(A_sub)
                KG_tf32 = truncate_to_tf32(KG_js)
                raw = A_tf32 @ KG_tf32  # [16, 32]

                q_scale = torch.exp2(g_slice[is_ * SUB_T : (is_ + 1) * SUB_T, :] - g_norm)
                dQ[is_ * SUB_T : (is_ + 1) * SUB_T, d_start:d_end] += raw * q_scale

        # Pass 2: dK_lower = tril(dAkk, 0) × KG (including diagonal)
        for is_ in range(num_sub):
            for js in range(is_ + 1):
                g_norm = g_slice[js * SUB_T, :]
                KG_js = k_slice[js * SUB_T : (js + 1) * SUB_T, :] * torch.exp2(
                    g_norm - g_slice[js * SUB_T : (js + 1) * SUB_T, :]
                )

                A_sub = dAkk[is_ * SUB_T : (is_ + 1) * SUB_T, js * SUB_T : (js + 1) * SUB_T].clone()
                if is_ == js:
                    # Causal mask: i >= j (zero upper triangle)
                    mask = torch.triu(torch.ones(SUB_T, SUB_T, device=q.device), diagonal=1).bool()
                    A_sub[mask] = 0.0

                A_tf32 = truncate_to_tf32(A_sub)
                KG_tf32 = truncate_to_tf32(KG_js)
                raw = A_tf32 @ KG_tf32

                q_scale = torch.exp2(g_slice[is_ * SUB_T : (is_ + 1) * SUB_T, :] - g_norm)
                dK[is_ * SUB_T : (is_ + 1) * SUB_T, d_start:d_end] += raw * q_scale

        # Pass 3: dK_upper_qk = triu(dAqk)^T × QG
        for js in range(num_sub):
            for is_ in range(js, num_sub):
                g_norm = g_slice[is_ * SUB_T, :]
                QG_is = q_slice[is_ * SUB_T : (is_ + 1) * SUB_T, :] * torch.exp2(
                    g_slice[is_ * SUB_T : (is_ + 1) * SUB_T, :] - g_norm
                )

                A_sub = dAqk[is_ * SUB_T : (is_ + 1) * SUB_T, js * SUB_T : (js + 1) * SUB_T].clone()
                if is_ == js:
                    mask = torch.triu(torch.ones(SUB_T, SUB_T, device=q.device), diagonal=1).bool()
                    A_sub[mask] = 0.0
                A_sub_T = A_sub.T

                A_tf32 = truncate_to_tf32(A_sub_T)
                QG_tf32 = truncate_to_tf32(QG_is)
                raw = A_tf32 @ QG_tf32

                k_scale = torch.exp2(torch.clamp(g_norm - g_slice[js * SUB_T : (js + 1) * SUB_T, :], min=-126.0, max=126.0))
                dK[js * SUB_T : (js + 1) * SUB_T, d_start:d_end] += raw * k_scale

        # Pass 4: dK_upper_kk = triu(dAkk, 0)^T × KBG (including diagonal)
        for js in range(num_sub):
            for is_ in range(js, num_sub):
                g_norm = g_slice[is_ * SUB_T, :]
                KBG_is = (
                    k_slice[is_ * SUB_T : (is_ + 1) * SUB_T, :]
                    * beta[is_ * SUB_T : (is_ + 1) * SUB_T, None]
                    * torch.exp2(g_slice[is_ * SUB_T : (is_ + 1) * SUB_T, :] - g_norm)
                )

                A_sub = dAkk[is_ * SUB_T : (is_ + 1) * SUB_T, js * SUB_T : (js + 1) * SUB_T].clone()
                if is_ == js:
                    # Causal mask: keep j >= i → tril of original dAkk
                    # After transpose: keep k >= m
                    mask = torch.triu(torch.ones(SUB_T, SUB_T, device=q.device), diagonal=1).bool()
                    A_sub[mask] = 0.0  # zero strict upper → keep tril(dAkk, 0)
                A_sub_T = A_sub.T

                A_tf32 = truncate_to_tf32(A_sub_T)
                KBG_tf32 = truncate_to_tf32(KBG_is)
                raw = A_tf32 @ KBG_tf32

                k_scale = torch.exp2(torch.clamp(g_norm - g_slice[js * SUB_T : (js + 1) * SUB_T, :], min=-126.0, max=126.0))
                dK[js * SUB_T : (js + 1) * SUB_T, d_start:d_end] += raw * k_scale

    return dQ, dK


@pytest.mark.parametrize(
    "B, T, H, D",
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1-D128"),
        pytest.param(1, 64, 2, 128, id="B1-T64-H2-D128"),
        pytest.param(2, 128, 2, 128, id="B2-T128-H2-D128"),
    ],
)
def test_bwd_intra_sm90_phase3_mma_output(B, T, H, D):
    """Phase 3: Verify MMA 4-pass output (dQ_intra, dK_intra) against Python reference."""
    torch.manual_seed(42)
    device = "cuda"
    BT = 64

    total_q_len = B * T
    q = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.1
    # Make G monotonically decreasing within each chunk
    for b in range(B):
        for c in range((T + BT - 1) // BT):
            start = b * T + c * BT
            end = min(start + BT, b * T + T)
            if start < end:
                g[start:end] = g[start:end].cumsum(dim=0).neg()

    # Set beta=1 so beta scaling is a no-op (Phase 3 tests MMA correctness, not epilogue)
    beta = torch.ones(total_q_len, H, dtype=torch.float32, device=device)
    dAqk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01
    dAkk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01

    # Set inter-chunk gradients to zero so output = intra only
    dq = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)
    chunk_indices = make_chunk_indices(cu_seqlens, BT)

    dq_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db_out = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    from cula.cudac import chunk_kda_bwd_intra_sm90

    chunk_kda_bwd_intra_sm90(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        BT,
    )
    torch.cuda.synchronize()

    # Basic sanity: no NaN/Inf
    assert not torch.isnan(dq_out).any(), "dq_out contains NaN"
    assert not torch.isnan(dk_out).any(), "dk_out contains NaN"
    assert not torch.isinf(dq_out).any(), "dq_out contains Inf"
    assert not torch.isinf(dk_out).any(), "dk_out contains Inf"

    # Verify per-chunk, per-head against Python reference
    num_chunks = T // BT
    for b_idx in range(B):
        for c in range(num_chunks):
            for h in range(H):
                start = b_idx * T + c * BT
                end = start + BT

                q_ch = q[start:end, h, :]  # [64, D] bf16
                k_ch = k[start:end, h, :]  # [64, D] bf16
                g_ch = g[start:end, h, :]  # [64, D] fp32
                beta_ch = beta[start:end, h]  # [64] fp32

                # Reconstruct dAqk/dAkk for this chunk [64, 64]
                dAqk_ch = dAqk[start:end, h, :]  # [64, BT=64]
                dAkk_ch = dAkk[start:end, h, :]  # [64, BT=64]

                dq_ref, dk_ref = compute_dq_dk_intra_ref(q_ch, k_ch, g_ch, beta_ch, dAqk_ch, dAkk_ch)

                dq_cuda = dq_out[start:end, h, :]  # [64, D]
                dk_cuda = dk_out[start:end, h, :]  # [64, D]

                torch.testing.assert_close(
                    dq_cuda,
                    dq_ref,
                    atol=5e-3,
                    rtol=1e-3,
                    msg=f"dQ mismatch: b={b_idx}, c={c}, h={h}",
                )
                torch.testing.assert_close(
                    dk_cuda,
                    dk_ref,
                    atol=5e-3,
                    rtol=1e-3,
                    msg=f"dK mismatch: b={b_idx}, c={c}, h={h}",
                )

    print(f"[PASS] Phase 3 MMA output test: B={B}, T={T}, H={H}, D={D}")


# =====================================================================
# Phase 3b: Ground-truth verification using kda_bwd_intra_kernel_ref
# =====================================================================


def compute_dq_dk_intra_ref_gnorm(q, k, g, beta, dAqk, dAkk, BT=64, BC=16):
    """Compute dQ_intra and dK_intra using G_norm factoring (no TF32).

    Adapted from kda_bwd_intra_kernel_ref (the verified FLA reference).
    Uses per-subchunk G_norm for numerical stability. G_norm cancels exactly
    so this produces the same result as the direct formula exp2(G[i]-G[j]).

    Args:
        q: [T, D] bf16 (one chunk, one head)
        k: [T, D] bf16
        g: [T, D] fp32
        beta: [T] fp32
        dAqk: [T, T] fp32
        dAkk: [T, T] fp32
    Returns:
        dQ_intra, dK_intra: [T, D] fp32
        dK_intra = dK_lower + dK_upper (before beta scaling on dK_lower)
    """
    T, D = q.shape
    q_f = q.float()
    k_f = k.float()
    BK = 32
    NC = BT // BC
    NK = (D + BK - 1) // BK

    dQ = torch.zeros(T, D, dtype=torch.float32, device=q.device)
    dK = torch.zeros(T, D, dtype=torch.float32, device=q.device)

    for i_k in range(NK):
        d_s = i_k * BK
        d_e = min(d_s + BK, D)
        for i_i in range(NC):
            i_ti = i_i * BC
            b_g = g[i_ti : i_ti + BC, d_s:d_e]

            b_dq2 = torch.zeros(BC, d_e - d_s, dtype=torch.float32, device=q.device)
            b_dk2 = torch.zeros(BC, d_e - d_s, dtype=torch.float32, device=q.device)

            # ── Off-diagonal lower: j < i_i ──
            if i_i > 0:
                b_gn = g[i_ti, d_s:d_e]  # G_norm = start of output subchunk
                for j in range(i_i):
                    i_tj = j * BC
                    b_k_j = k_f[i_tj : i_tj + BC, d_s:d_e]
                    b_gk = g[i_tj : i_tj + BC, d_s:d_e]
                    b_kg = b_k_j * torch.exp2(b_gn[None, :] - b_gk)

                    b_dAqk_ij = dAqk[i_ti : i_ti + BC, j * BC : (j + 1) * BC]
                    b_dAkk_ij = dAkk[i_ti : i_ti + BC, j * BC : (j + 1) * BC]
                    b_dq2 += b_dAqk_ij @ b_kg
                    b_dk2 += b_dAkk_ij @ b_kg

                b_gqn = torch.exp2(b_g - b_gn[None, :])
                b_dq2 *= b_gqn
                b_dk2 *= b_gqn

            # ── Diagonal: is == js ──
            b_gn_mid = g[i_ti + min(BC // 2, T - i_ti - 1), d_s:d_e]
            b_dAqk_diag = dAqk[i_ti : i_ti + BC, i_i * BC : (i_i + 1) * BC]
            b_dAkk_diag = dAkk[i_ti : i_ti + BC, i_i * BC : (i_i + 1) * BC]

            o_i = torch.arange(BC, device=q.device)
            m_lower = o_i[:, None] >= o_i[None, :]  # tril mask for dQ/dK_lower
            b_dAqk_masked = torch.where(m_lower, b_dAqk_diag, 0.0)
            b_dAkk_masked = torch.where(m_lower, b_dAkk_diag, 0.0)

            b_g_rel = b_g - b_gn_mid[None, :]
            b_k_scaled = k_f[i_ti : i_ti + BC, d_s:d_e] * torch.exp2(-b_g_rel)
            b_row_scale = torch.exp2(b_g_rel)

            b_dq2 += (b_dAqk_masked @ b_k_scaled) * b_row_scale
            b_dk2 += (b_dAkk_masked @ b_k_scaled) * b_row_scale

            dQ[i_ti : i_ti + BC, d_s:d_e] += b_dq2
            dK[i_ti : i_ti + BC, d_s:d_e] += b_dk2

            # ── Off-diagonal upper: j > i_i ──
            b_dkt = torch.zeros(BC, d_e - d_s, dtype=torch.float32, device=q.device)
            if i_i < NC - 1:
                b_gn_next = g[i_ti + BC, d_s:d_e]
                for j in range(i_i + 1, NC):
                    i_tj = j * BC
                    b_q_j = q_f[i_tj : i_tj + BC, d_s:d_e]
                    b_kb_j = k_f[i_tj : i_tj + BC, d_s:d_e] * beta[i_tj : i_tj + BC, None]
                    b_gk_j = g[i_tj : i_tj + BC, d_s:d_e]

                    b_gkn = torch.exp2(b_gk_j - b_gn_next[None, :])
                    b_qg = b_q_j * b_gkn
                    b_kbg = b_kb_j * b_gkn

                    b_dAqk_T = dAqk[i_tj : i_tj + BC, i_i * BC : (i_i + 1) * BC].T
                    b_dAkk_T = dAkk[i_tj : i_tj + BC, i_i * BC : (i_i + 1) * BC].T
                    b_dkt += b_dAqk_T @ b_qg
                    b_dkt += b_dAkk_T @ b_kbg

                b_dkt *= torch.exp2(b_gn_next[None, :] - b_g)

            # ── Upper diagonal ──
            b_dAqk_T_diag = dAqk[i_ti : i_ti + BC, i_i * BC : (i_i + 1) * BC].T
            b_dAkk_T_diag = dAkk[i_ti : i_ti + BC, i_i * BC : (i_i + 1) * BC].T
            m_upper = o_i[:, None] <= o_i[None, :]  # triu mask
            b_dAqk_T_masked = torch.where(m_upper, b_dAqk_T_diag, 0.0)
            b_dAkk_T_masked = torch.where(m_upper, b_dAkk_T_diag, 0.0)

            b_q_exp = q_f[i_ti : i_ti + BC, d_s:d_e] * b_row_scale
            b_kb_exp = k_f[i_ti : i_ti + BC, d_s:d_e] * beta[i_ti : i_ti + BC, None] * b_row_scale
            tmp_dkt = (b_dAqk_T_masked @ b_q_exp) + (b_dAkk_T_masked @ b_kb_exp)
            tmp_dkt *= torch.exp2(-b_g_rel)

            b_dkt += tmp_dkt
            dK[i_ti : i_ti + BC, d_s:d_e] += b_dkt

    return dQ, dK


@pytest.mark.parametrize(
    "B, T, H, D",
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1-D128"),
        pytest.param(1, 64, 2, 128, id="B1-T64-H2-D128"),
    ],
)
def test_bwd_intra_sm90_phase3b_ground_truth(B, T, H, D):
    """Phase 3b: Verify kernel output against G_norm-based reference (no TF32).

    Uses kda_bwd_intra_kernel_ref logic as ground truth. Since the reference
    has no TF32 truncation, tolerance is wider to account for mma.sync error.
    """
    torch.manual_seed(42)
    device = "cuda"
    BT = 64

    total_q_len = B * T
    q = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.1
    for b in range(B):
        for c in range((T + BT - 1) // BT):
            start = b * T + c * BT
            end = min(start + BT, b * T + T)
            if start < end:
                g[start:end] = g[start:end].cumsum(dim=0).neg()

    beta = torch.ones(total_q_len, H, dtype=torch.float32, device=device)
    dAqk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01
    dAkk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01

    dq = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)
    chunk_indices = make_chunk_indices(cu_seqlens, BT)

    dq_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db_out = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    from cula.cudac import chunk_kda_bwd_intra_sm90

    chunk_kda_bwd_intra_sm90(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        BT,
    )
    torch.cuda.synchronize()

    num_chunks = T // BT
    for b_idx in range(B):
        for c in range(num_chunks):
            for h in range(H):
                start = b_idx * T + c * BT
                end = start + BT

                q_ch = q[start:end, h, :]
                k_ch = k[start:end, h, :]
                g_ch = g[start:end, h, :]
                beta_ch = beta[start:end, h]
                dAqk_ch = dAqk[start:end, h, :]
                dAkk_ch = dAkk[start:end, h, :]

                dq_gt, dk_gt = compute_dq_dk_intra_ref_gnorm(q_ch, k_ch, g_ch, beta_ch, dAqk_ch, dAkk_ch)

                dq_cuda = dq_out[start:end, h, :]
                dk_cuda = dk_out[start:end, h, :]

                # Tolerance: TF32 mma.sync truncation
                torch.testing.assert_close(
                    dq_cuda,
                    dq_gt,
                    atol=5e-3,
                    rtol=1e-3,
                    msg=f"dQ ref mismatch: b={b_idx}, c={c}, h={h}",
                )
                torch.testing.assert_close(
                    dk_cuda,
                    dk_gt,
                    atol=5e-3,
                    rtol=1e-3,
                    msg=f"dK ref mismatch: b={b_idx}, c={c}, h={h}",
                )

    print(f"[PASS] Phase 3b ground-truth test: B={B}, T={T}, H={H}, D={D}")


# =====================================================================
# Phase 4: Full Epilogue (dQ / dK / dB / dG) end-to-end
# =====================================================================


def compute_epilogue_ref(q_ch, k_ch, g_ch, beta_ch, dAqk_ch, dAkk_ch, dq_inter_ch, dk_inter_ch, dg_inter_ch, db_inter_ch):
    """Per-chunk, per-head full backward intra reference (fp32).

    Args:
        q_ch: [BT, D] bf16  (one chunk, one head)
        k_ch: [BT, D] bf16
        g_ch: [BT, D] fp32
        beta_ch: [BT] fp32
        dAqk_ch: [BT, BT] fp32
        dAkk_ch: [BT, BT] fp32
        dq_inter_ch: [BT, D] fp32
        dk_inter_ch: [BT, D] fp32
        dg_inter_ch: [BT, D] fp32
        db_inter_ch: [BT] fp32

    Returns:
        dq_out, dk_out, dg_out: [BT, D] fp32
        db_out: [BT] fp32
    """
    BT, D = q_ch.shape
    q = q_ch.float()
    k = k_ch.float()
    g = g_ch  # already fp32
    beta = beta_ch  # [BT]

    # Mask: i >= j (lower triangle including diagonal)
    mask = torch.tril(torch.ones(BT, BT, device=q.device))  # [BT, BT]

    dAqk_masked = dAqk_ch * mask  # [BT, BT]
    dAkk_masked = dAkk_ch * mask  # [BT, BT]

    # gate_3d[i, j, d] = exp2(g[i,d] - g[j,d])
    gate_3d = torch.exp2(g.unsqueeze(1) - g.unsqueeze(0))  # [BT, BT, D]

    # Coefficients: dA_masked * gate  → [BT, BT, D]
    coeff_qk = dAqk_masked.unsqueeze(-1) * gate_3d
    coeff_kk = dAkk_masked.unsqueeze(-1) * gate_3d

    # dQ_intra[i, d] = Σ_j coeff_qk[i,j,d] * K[j,d]
    dq_intra = torch.einsum("ijd,jd->id", coeff_qk, k)

    # dK_lower[i, d] = Σ_j coeff_kk[i,j,d] * K[j,d]
    dk_lower = torch.einsum("ijd,jd->id", coeff_kk, k)

    # dK_upper_qk[j, d] = Σ_{i>=j} dAqk[i,j]*gate[i,j,d]*Q[i,d]
    dk_upper_qk = torch.einsum("ijd,id->jd", coeff_qk, q)

    # dK_upper_kk[j, d] = Σ_{i>=j} dAkk[i,j]*gate[i,j,d]*K[i,d]*beta[i]
    dk_upper_kk = torch.einsum("ijd,id->jd", coeff_kk, k * beta.unsqueeze(-1))

    dk_upper = dk_upper_qk + dk_upper_kk

    # Epilogue
    # dB[row] = Σ_d dk_lower[row,d] * K[row,d]  (BEFORE beta scaling)
    db_intra = (dk_lower * k).sum(dim=-1)  # [BT]

    # Beta scaling on dK_lower
    dk_lower_beta = dk_lower * beta.unsqueeze(-1)

    # Final outputs
    dq_out = dq_inter_ch + dq_intra
    dk_out = dk_inter_ch + dk_lower_beta + dk_upper
    db_out = db_intra + db_inter_ch
    dg_out = dg_inter_ch + q * dq_intra + (dk_lower_beta - dk_upper) * k

    return dq_out, dk_out, db_out, dg_out


@pytest.mark.parametrize(
    "B, T, H, D",
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1-D128"),
        pytest.param(1, 64, 2, 128, id="B1-T64-H2-D128"),
        pytest.param(2, 128, 2, 128, id="B2-T128-H2-D128"),
    ],
)
def test_bwd_intra_sm90_phase4_epilogue(B, T, H, D):
    """Phase 4: Full epilogue end-to-end — dQ, dK, dB, dG vs Python reference."""
    torch.manual_seed(42)
    device = "cuda"
    BT = 64

    total_q_len = B * T
    q = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_q_len, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.1
    for b in range(B):
        for c in range((T + BT - 1) // BT):
            start = b * T + c * BT
            end = min(start + BT, b * T + T)
            if start < end:
                g[start:end] = g[start:end].cumsum(dim=0).neg()

    beta = torch.rand(total_q_len, H, dtype=torch.float32, device=device) * 0.5 + 0.5
    dAqk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01
    dAkk = torch.randn(total_q_len, H, BT, dtype=torch.float32, device=device) * 0.01

    # Inter-chunk inputs (non-zero to test additive combination)
    dq = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01
    dk = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01
    db = torch.randn(total_q_len, H, dtype=torch.float32, device=device) * 0.01
    dg = torch.randn(total_q_len, H, D, dtype=torch.float32, device=device) * 0.01

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)
    chunk_indices = make_chunk_indices(cu_seqlens, BT)

    dq_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    db_out = torch.zeros(total_q_len, H, dtype=torch.float32, device=device)
    dg_out = torch.zeros(total_q_len, H, D, dtype=torch.float32, device=device)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    from cula.cudac import chunk_kda_bwd_intra_sm90

    chunk_kda_bwd_intra_sm90(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        BT,
    )
    torch.cuda.synchronize()

    # Compare per chunk, per head
    num_chunks = T // BT
    for b_idx in range(B):
        for c in range(num_chunks):
            for h in range(H):
                start = b_idx * T + c * BT
                end = start + BT

                dq_gt, dk_gt, db_gt, dg_gt = compute_epilogue_ref(
                    q[start:end, h, :],
                    k[start:end, h, :],
                    g[start:end, h, :],
                    beta[start:end, h],
                    dAqk[start:end, h, :],
                    dAkk[start:end, h, :],
                    dq[start:end, h, :],
                    dk[start:end, h, :],
                    dg[start:end, h, :],
                    db[start:end, h],
                )

                tag = f"b={b_idx}, c={c}, h={h}"
                torch.testing.assert_close(
                    dq_out[start:end, h, :],
                    dq_gt,
                    atol=8e-3,
                    rtol=1e-3,
                    msg=f"dQ mismatch: {tag}",
                )
                torch.testing.assert_close(
                    dk_out[start:end, h, :],
                    dk_gt,
                    atol=8e-3,
                    rtol=1e-3,
                    msg=f"dK mismatch: {tag}",
                )
                torch.testing.assert_close(
                    db_out[start:end, h],
                    db_gt,
                    atol=0.02,
                    rtol=1e-2,
                    msg=f"dB mismatch: {tag}",
                )
                torch.testing.assert_close(
                    dg_out[start:end, h, :],
                    dg_gt,
                    atol=0.02,
                    rtol=1e-2,
                    msg=f"dG mismatch: {tag}",
                )

    print(f"[PASS] Phase 4 epilogue test: B={B}, T={T}, H={H}, D={D}")
