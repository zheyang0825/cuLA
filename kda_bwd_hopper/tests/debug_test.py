"""Debug test to isolate the correctness bug in kda_bwd_intra SM90 kernel."""
import torch
import os, sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "kda-backward-internal-main", "third_party", "flash-linear-attention"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "kda-backward-internal-main"))

from kda.interface import kda_bwd_intra
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra

torch.backends.cuda.matmul.allow_tf32 = True

def run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, chunk_size):
    dq_out = torch.empty_like(dq, dtype=torch.bfloat16)
    dk_out = torch.empty_like(dk, dtype=torch.bfloat16)
    db_out = torch.empty_like(db, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    return kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                         cu_seqlens, chunk_indices, dq_out, dk_out, db_out, dg_out,
                         chunk_size, tile_counter)

def run_fla(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, chunk_size):
    return chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                               cu_seqlens, chunk_indices, chunk_size, True)

def compare(name, ref, test, tol=0.008):
    ref_f = ref.float()
    test_f = test.float()
    diff = (ref_f - test_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    error_rate = (diff > tol).float().mean().item() * 100
    ref_norm = ref_f.abs().mean().item()
    test_norm = test_f.abs().mean().item()
    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"error_rate={error_rate:.2f}% (tol={tol}), "
          f"ref_mean_abs={ref_norm:.6f}, test_mean_abs={test_norm:.6f}")
    if error_rate > 0 and ref_f.numel() <= 200:
        # Print first few mismatches
        mask = diff > tol
        idx = mask.nonzero()[:5]
        for i in range(min(5, len(idx))):
            pos = tuple(idx[i].tolist())
            print(f"    mismatch at {pos}: ref={ref_f[pos].item():.6f}, test={test_f[pos].item():.6f}")

def create_data(B, T, H, K, seed=42, varlen=False):
    torch.manual_seed(seed)
    dev = 'cuda'
    BT = 64

    if varlen:
        import random
        random.seed(seed)
        cu_seqlens = torch.zeros(B + 1, device=dev, dtype=torch.int32)
        TILE_NUM = 0
        for i in range(1, B + 1):
            seq_len = min(random.normalvariate(T, T / 2), T*2)
            cu_seqlens[i] = cu_seqlens[i - 1] + int(seq_len)
            TILE_NUM += int((int(seq_len) + 64 - 1) // 64)
    else:
        cu_seqlens = torch.arange(0, (B+1)*T, T, device=dev, dtype=torch.int32)
        TILE_NUM = B * ((T + 63) // 64)

    chunk_indices = torch.zeros(TILE_NUM, 2, device=dev, dtype=torch.int32)
    acc = 0
    for i in range(B):
        sl = (cu_seqlens[i+1] - cu_seqlens[i]).item()
        nt = (sl + 63) // 64
        for j in range(nt):
            chunk_indices[acc + j, 0] = i
            chunk_indices[acc + j, 1] = j
        acc += nt

    total_len = cu_seqlens[-1].item()
    q = torch.randn(1, total_len, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, K, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, total_len, H, K, device=dev, dtype=torch.float32) / 10
    beta = torch.randn(1, total_len, H, device=dev, dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, BT, device=dev, dtype=torch.float32)
    dAkk = torch.randn(1, total_len, H, BT, device=dev, dtype=torch.float32)
    dq = torch.randn(1, total_len, H, K, device=dev, dtype=torch.float32)
    dk = torch.randn(1, total_len, H, K, device=dev, dtype=torch.float32)
    db = torch.randn(1, total_len, H, device=dev, dtype=torch.float32)
    dg = torch.randn(1, total_len, H, K, device=dev, dtype=torch.float32)

    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT

# ==== TEST 1: NULL TEST (dA=0) ====
print("=" * 60)
print("TEST 1: NULL test (dAqk=0, dAkk=0)")
print("Expected: dq_out = bf16(dq_in), dk_out = bf16(dk_in), db_out = db_in, dg_out = dg_in")
print("=" * 60)

q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT = \
    create_data(B=1, T=64, H=1, K=128, seed=0)

# Zero out dA matrices
dAqk.zero_()
dAkk.zero_()

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)

# Expected outputs
dq_expected = dq.to(torch.bfloat16)
dk_expected = dk.to(torch.bfloat16)
db_expected = db
dg_expected = dg

compare("dq_null", dq_expected, dq_c, tol=0.0)
compare("dk_null", dk_expected, dk_c, tol=0.0)
compare("db_null", db_expected, db_c, tol=0.0)
compare("dg_null", dg_expected, dg_c, tol=0.0)

# ==== TEST 2: Small full test (B=1, T=64, H=1) ====
print("\n" + "=" * 60)
print("TEST 2: Small test (B=1, T=64, H=1, K=128)")
print("=" * 60)

q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT = \
    create_data(B=1, T=64, H=1, K=128, seed=42)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)

compare("dq", dq_f, dq_c, tol=0.008)
compare("dk", dk_f, dk_c, tol=0.008)
compare("db", db_f, db_c, tol=0.02)
compare("dg", dg_f, dg_c, tol=0.02)

# Print some values to understand the pattern
print("\n  First 8 elements of dq (head=0, feat=0..7, token=0):")
print(f"    FLA:  {dq_f[0, 0, 0, :8].tolist()}")
print(f"    CUDA: {dq_c[0, 0, 0, :8].tolist()}")
print(f"    dq_in:{dq[0, 0, 0, :8].tolist()}")

# ==== TEST 3: Isolate forward diagonal only (T=16, single sub-chunk) ====
print("\n" + "=" * 60)
print("TEST 3: Single sub-chunk (B=1, T=16, H=1, K=128) - diagonal only")
print("=" * 60)

q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT = \
    create_data(B=1, T=16, H=1, K=128, seed=42)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)

compare("dq", dq_f, dq_c, tol=0.008)
compare("dk", dk_f, dk_c, tol=0.008)
compare("db", db_f, db_c, tol=0.02)
compare("dg", dg_f, dg_c, tol=0.02)

# ==== TEST 4: Two sub-chunks (T=32) - has one off-diagonal ====
print("\n" + "=" * 60)
print("TEST 4: Two sub-chunks (B=1, T=32, H=1, K=128)")
print("=" * 60)

q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT = \
    create_data(B=1, T=32, H=1, K=128, seed=42)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)

compare("dq", dq_f, dq_c, tol=0.008)
compare("dk", dk_f, dk_c, tol=0.008)
compare("db", db_f, db_c, tol=0.02)
compare("dg", dg_f, dg_c, tol=0.02)

# ==== TEST 5: Multi-head (B=1, T=64, H=4) ====
print("\n" + "=" * 60)
print("TEST 5: Multi-head (B=1, T=64, H=4, K=128)")
print("=" * 60)

q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT = \
    create_data(B=1, T=64, H=4, K=128, seed=42)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)

compare("dq", dq_f, dq_c, tol=0.008)
compare("dk", dk_f, dk_c, tol=0.008)
compare("db", db_f, db_c, tol=0.02)
compare("dg", dg_f, dg_c, tol=0.02)

print("\nDone!")
