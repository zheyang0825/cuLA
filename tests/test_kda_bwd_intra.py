"""
Official-style test for kda_bwd_intra SM90 CUTLASS kernel.

Compares our CUDA kernel output against FLA Triton reference (chunk_kda_bwd_intra).
Post-processes db_out (sum over NK + add upstream) before comparison.

FLA assert_close uses error_ratio = RMSE(diff) / RMS(ref), not absolute tolerance.
"""
import random
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cula.cudac as C
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as fla_chunk_kda_bwd_intra


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = x.detach().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def assert_close(name, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    status = "PASS" if (abs_atol <= err_atol or error_rate < ratio) else "FAIL"
    print(f"  [{status}] {name:>8}  abs_max={abs_atol:.6f}  err_ratio={error_rate:.6f}  (limit={ratio})")
    assert abs_atol <= err_atol or error_rate < ratio, (
        f"{name} abs_max={abs_atol:.6f} err_ratio={error_rate:.6f} > {ratio}"
    )


def generate_data(seed=42, B=4, T=128, H=4, K=128, varlen=False):
    torch.manual_seed(seed)
    random.seed(seed)
    BT = 64

    cu_seqlens = torch.zeros(B + 1, device="cuda", dtype=torch.int32)
    TILE_NUM = 0
    if varlen:
        for i in range(1, B + 1):
            seq_len = min(max(int(random.normalvariate(T, T / 2)), BT), T * 2)
            cu_seqlens[i] = cu_seqlens[i - 1] + seq_len
            TILE_NUM += (seq_len + BT - 1) // BT
    else:
        for i in range(B + 1):
            cu_seqlens[i] = i * T
        TILE_NUM = B * ((T + BT - 1) // BT)

    chunk_indices = torch.zeros(TILE_NUM, 2, device="cuda", dtype=torch.int32)
    acc = 0
    for i in range(B):
        seq_len = int(cu_seqlens[i + 1] - cu_seqlens[i])
        ntiles = (seq_len + BT - 1) // BT
        for j in range(ntiles):
            chunk_indices[acc + j, 0] = i
            chunk_indices[acc + j, 1] = j
        acc += ntiles

    total_len = int(cu_seqlens[-1])
    NK = K // 32

    q = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32) / 10
    beta = torch.randn(1, total_len, H, device="cuda", dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dAkk = torch.randn(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dq = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32)
    dk = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32)
    db = torch.randn(1, total_len, H, device="cuda", dtype=torch.float32)
    dg = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32)

    # Output tensors for our CUDA kernel
    dq_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    dk_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    db_out = torch.zeros(NK, 1, total_len, H, device="cuda", dtype=torch.float32)
    dg_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)

    return dict(
        q=q, k=k, g=g, beta=beta,
        dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        dq_out=dq_out, dk_out=dk_out, db_out=db_out, dg_out=dg_out,
        chunk_size=BT, B=B, T=T, H=H, K=K, NK=NK,
    )


def run_cuda(data):
    """Run our CUDA kernel and return post-processed outputs matching FLA format."""
    # Reset outputs
    data["dq_out"].zero_()
    data["dk_out"].zero_()
    data["db_out"].zero_()
    data["dg_out"].zero_()

    C.chunk_kda_bwd_intra_cuda(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["dq_out"], data["dk_out"], data["db_out"], data["dg_out"],
        data["chunk_size"],
    )
    torch.cuda.synchronize()

    dq_out = data["dq_out"]
    dk_out = data["dk_out"]
    # Post-process db: sum over NK dimension + add upstream db
    db_out = data["db_out"].sum(0).add_(data["db"])
    # dg_out already includes upstream dg in our kernel
    dg_out = data["dg_out"]

    return dq_out, dk_out, db_out, dg_out


def run_fla(data):
    """Run FLA Triton reference and return outputs."""
    dq, dk, db, dg = fla_chunk_kda_bwd_intra(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["chunk_size"], safe_gate=True,
    )
    return dq, dk, db, dg


def test_correctness(label, data, dq_ratio=0.02, dk_ratio=0.02, db_ratio=0.04, dg_ratio=0.04):
    """Compare CUDA output vs FLA Triton reference."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    dq_cuda, dk_cuda, db_cuda, dg_cuda = run_cuda(data)
    dq_fla, dk_fla, db_fla, dg_fla = run_fla(data)

    assert_close("dq", dq_fla, dq_cuda, dq_ratio)
    assert_close("dk", dk_fla, dk_cuda, dk_ratio)
    assert_close("db", db_fla, db_cuda, db_ratio)
    assert_close("dg", dg_fla, dg_cuda, dg_ratio)

    print(f"  [PASS] {label}")
    return True


def test_determinism(data, num_runs=20):
    """Check that kernel produces identical results across runs."""
    print(f"\n{'=' * 60}")
    print(f"  Determinism check ({num_runs} runs)")
    print(f"{'=' * 60}")

    dq_base, dk_base, db_base, dg_base = run_cuda(data)
    for i in range(num_runs):
        dq, dk, db, dg = run_cuda(data)
        assert torch.equal(dq, dq_base), f"dq not deterministic at run {i}"
        assert torch.equal(dk, dk_base), f"dk not deterministic at run {i}"
        # db involves sum reduction so use allclose
        assert torch.allclose(db, db_base, atol=0, rtol=0), f"db not deterministic at run {i}"
        assert torch.equal(dg, dg_base), f"dg not deterministic at run {i}"
    print(f"  [PASS] Deterministic over {num_runs} runs")
    return True


if __name__ == "__main__":
    ok = True

    # Test 1: Fixed-length, small
    d = generate_data(seed=42, B=2, T=128, H=2, K=128, varlen=False)
    ok &= test_correctness("Fixed B=2 T=128 H=2 K=128", d)

    # Test 2: Variable-length
    d = generate_data(seed=123, B=4, T=96, H=2, K=128, varlen=True)
    ok &= test_correctness("Varlen B=4 T=96 H=2 K=128", d)

    # Test 3: Larger, production-like
    d = generate_data(seed=42, B=10, T=800, H=96, K=128, varlen=True)
    ok &= test_correctness("Varlen B=10 T=800 H=96 K=128", d)

    # Test 4: Determinism on production-like data
    ok &= test_determinism(d, num_runs=10)

    print(f"\n{'=' * 60}")
    print(f"  Overall: {'ALL PASSED' if ok else 'SOME FAILED'}")
    print(f"{'=' * 60}")
    exit(0 if ok else 1)
