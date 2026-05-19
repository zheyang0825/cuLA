"""Targeted debug: isolate the diagonal MMA computation."""
import torch
import os, sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "kda-backward-internal-main", "third_party", "flash-linear-attention"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "kda-backward-internal-main"))

from kda.interface import kda_bwd_intra
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra

torch.backends.cuda.matmul.allow_tf32 = True

def run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT):
    dq_out = torch.empty_like(dq, dtype=torch.bfloat16)
    dk_out = torch.empty_like(dk, dtype=torch.bfloat16)
    db_out = torch.empty_like(db, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    tc = torch.zeros(1, dtype=torch.int32, device=q.device)
    return kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                         cu_seqlens, chunk_indices, dq_out, dk_out, db_out, dg_out, BT, tc)

def run_fla(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT):
    return chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                               cu_seqlens, chunk_indices, BT, True)

dev = 'cuda'
BT = 64
T = 16  # single sub-chunk
H = 1
K = 128

# ==== TEST A: Identity dA, ones k, zero g, zero inputs ====
print("=" * 60)
print("TEST A: Identity dA, ones k, zero g (B=1, T=16, H=1)")
print("Expected: dq_intra = ones(16, 128), dq_out = 1 + dq_in")
print("=" * 60)

cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
chunk_indices = torch.tensor([[0, 0]], device=dev, dtype=torch.int32)

q = torch.ones(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.ones(1, T, H, K, device=dev, dtype=torch.bfloat16)
g = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
beta = torch.zeros(1, T, H, device=dev, dtype=torch.bfloat16)

# Identity dA: dAqk[0, t, 0, c] = 1 if t==c else 0
dAqk = torch.zeros(1, T, H, BT, device=dev, dtype=torch.float32)
for t in range(T):
    dAqk[0, t, 0, t] = 1.0
dAkk = torch.zeros(1, T, H, BT, device=dev, dtype=torch.float32)

dq = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
dk = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
db = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
dg = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)

print(f"  dq_out shape: {dq_c.shape}")
print(f"  dq_out[0, :, 0, 0] (should be 1.0):")
print(f"    {dq_c[0, :, 0, 0].tolist()}")
print(f"  dq_out[0, 0, 0, :8] (should be 1.0):")
print(f"    {dq_c[0, 0, 0, :8].tolist()}")
print(f"  dq_out[0, 0, 0, 8:16]:")
print(f"    {dq_c[0, 0, 0, 8:16].tolist()}")

# Check FLA
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)
print(f"\n  FLA dq[0, :, 0, 0]:")
print(f"    {dq_f[0, :, 0, 0].tolist()}")
print(f"  FLA dq[0, 0, 0, :8]:")
print(f"    {dq_f[0, 0, 0, :8].tolist()}")

# ==== TEST B: Simple matmul check - dA = ones, k = sequential, g = 0 ====
print("\n" + "=" * 60)
print("TEST B: dA=lower_tri(ones), k=arange, g=0")
print("Expected: dq_intra[r,f] = sum_{c<=r} k[c,f] = sum_{c=0}^{r} (c*K+f+1)")
print("=" * 60)

# k[t, f] = t*K + f + 1 (1-indexed sequential)
k_vals = torch.arange(1, T*K+1, device=dev, dtype=torch.float32).reshape(1, T, 1, K)
k = k_vals.to(torch.bfloat16)
q = torch.ones(1, T, H, K, device=dev, dtype=torch.bfloat16)
g = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

# Lower triangular dA (all ones below and including diagonal)
dAqk = torch.zeros(1, T, H, BT, device=dev, dtype=torch.float32)
for r in range(T):
    for c in range(r+1):
        dAqk[0, r, 0, c] = 1.0
dAkk = torch.zeros_like(dAqk)

dq = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
dk = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
db = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
dg = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)

# Compute reference: dq_intra[r, f] = sum_{c=0}^{r} k_bf16[c, f]
k_f32 = k.float()
ref = torch.zeros(1, T, H, K, device=dev)
for r in range(T):
    ref[0, r, 0, :] = k_f32[0, :r+1, 0, :].sum(dim=0)

print(f"  dq_out[0, 0, 0, :8] (should be k[0,:8]): {dq_c[0, 0, 0, :8].tolist()}")
print(f"  ref   [0, 0, 0, :8]: {ref[0, 0, 0, :8].tolist()}")
print(f"  k_bf16[0, 0, 0, :8]: {k_f32[0, 0, 0, :8].tolist()}")
print()
print(f"  dq_out[0, 1, 0, :8] (should be k[0]+k[1]): {dq_c[0, 1, 0, :8].tolist()}")
print(f"  ref   [0, 1, 0, :8]: {ref[0, 1, 0, :8].tolist()}")
print()
print(f"  dq_out[0, 7, 0, :4]: {dq_c[0, 7, 0, :4].tolist()}")
print(f"  ref   [0, 7, 0, :4]: {ref[0, 7, 0, :4].tolist()}")
print()
print(f"  dq_out[0, 8, 0, :4]: {dq_c[0, 8, 0, :4].tolist()}")
print(f"  ref   [0, 8, 0, :4]: {ref[0, 8, 0, :4].tolist()}")
print()
print(f"  dq_out[0, 15, 0, :4]: {dq_c[0, 15, 0, :4].tolist()}")
print(f"  ref   [0, 15, 0, :4]: {ref[0, 15, 0, :4].tolist()}")

# ==== TEST C: Random data but zero g (removes gating complexity) ====
print("\n" + "=" * 60)
print("TEST C: Random data, g=0 (no gating)")
print("=" * 60)

torch.manual_seed(42)
q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
g = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
dAqk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)
dAkk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)
dq = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
dk = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
db = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
dg = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)

diff = (dq_c.float() - dq_f.float()).abs()
print(f"  dq: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
print(f"  dq error_rate: {(diff > 0.008).float().mean()*100:.2f}%")

# Print detailed comparison for token 0
print(f"\n  Token 0, features 0-7:")
print(f"    FLA:  {dq_f[0,0,0,:8].tolist()}")
print(f"    CUDA: {dq_c[0,0,0,:8].tolist()}")

# PyTorch reference for diagonal: lower_tri(dAqk[0:16, 0:16]) @ k[0:16, :]  (with g=0)
dA_block = dAqk[0, :T, 0, :T]  # [16, 16]
mask = torch.tril(torch.ones(T, T, device=dev))  # lower triangular
dA_masked = dA_block * mask  # [16, 16]

# With g=0 and safe_gate pivot gn_row=8: B = k * exp2(0-0) = k, scale = exp2(0-0) = 1
# Result = dA_masked @ k_bf16
k_f32 = k[0, :T, 0, :].float()  # [16, 128]
# But MMA uses TF32 rounding, so we need to simulate that
def to_tf32(x):
    """Simulate TF32 by truncating mantissa to 10 bits"""
    bits = x.view(torch.int32)
    bits = bits & 0xFFFFE000
    return bits.view(torch.float32)

ref_exact = dA_masked @ k_f32  # [16, 128]
ref_tf32_A = to_tf32(dA_masked)
ref_tf32_B = to_tf32(k_f32)
# TF32 matmul
ref_tf32 = ref_tf32_A @ ref_tf32_B

print(f"\n  PyTorch ref (exact): {ref_exact[0, :8].tolist()}")
print(f"  PyTorch ref (tf32):  {ref_tf32[0, :8].tolist()}")
print(f"  CUDA (dq_intra=dq_out since dq_in=0): {dq_c[0,0,0,:8].tolist()}")
print(f"  FLA:                 {dq_f[0,0,0,:8].tolist()}")

print("\nDone!")
