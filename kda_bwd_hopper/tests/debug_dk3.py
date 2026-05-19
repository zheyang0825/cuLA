"""Isolate dk error: test individual components."""
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
T = 16
H = 1
K = 128

torch.manual_seed(42)
cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
chunk_indices = torch.tensor([[0, 0]], device=dev, dtype=torch.int32)

q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
g = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
dAqk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)
dAkk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)

# Run with both dA, get dk_both
dq_z = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
dk_z = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
db_z = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
dg_z = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

_, dk_both_c, _, _ = run_cuda(q, k, g, beta, dAqk, dAkk, dq_z, dk_z, db_z, dg_z, cu_seqlens, chunk_indices, BT)
_, dk_both_f, _, _ = run_fla(q, k, g, beta, dAqk, dAkk, dq_z.clone(), dk_z.clone(), db_z.clone(), dg_z.clone(), cu_seqlens, chunk_indices, BT)

# Run with only dAqk
_, dk_qk_c, _, _ = run_cuda(q, k, g, beta, dAqk, torch.zeros_like(dAkk), dq_z, dk_z, db_z, dg_z, cu_seqlens, chunk_indices, BT)
_, dk_qk_f, _, _ = run_fla(q, k, g, beta, dAqk, torch.zeros_like(dAkk), dq_z.clone(), dk_z.clone(), db_z.clone(), dg_z.clone(), cu_seqlens, chunk_indices, BT)

# Run with only dAkk
_, dk_kk_c, _, _ = run_cuda(q, k, g, beta, torch.zeros_like(dAqk), dAkk, dq_z, dk_z, db_z, dg_z, cu_seqlens, chunk_indices, BT)
_, dk_kk_f, _, _ = run_fla(q, k, g, beta, torch.zeros_like(dAqk), dAkk, dq_z.clone(), dk_z.clone(), db_z.clone(), dg_z.clone(), cu_seqlens, chunk_indices, BT)

# Check individual components
d1 = (dk_qk_c.float() - dk_qk_f.float()).abs()
d2 = (dk_kk_c.float() - dk_kk_f.float()).abs()
d3 = (dk_both_c.float() - dk_both_f.float()).abs()

print(f"dAqk only: max_diff={d1.max():.6f} mean={d1.mean():.6f} err={(d1>0.008).float().mean()*100:.2f}%")
print(f"dAkk only: max_diff={d2.max():.6f} mean={d2.mean():.6f} err={(d2>0.008).float().mean()*100:.2f}%")
print(f"both:      max_diff={d3.max():.6f} mean={d3.mean():.6f} err={(d3>0.008).float().mean()*100:.2f}%")

# Check if dk_both = dk_qk + dk_kk (linear decomposition)
# For CUDA:
dk_sum_c = dk_qk_c.float() + dk_kk_c.float()
d4 = (dk_both_c.float() - dk_sum_c).abs()
print(f"\nCUDA dk_both vs dk_qk+dk_kk: max_diff={d4.max():.6f}")

# For FLA:
dk_sum_f = dk_qk_f.float() + dk_kk_f.float()
d5 = (dk_both_f.float() - dk_sum_f).abs()
print(f"FLA  dk_both vs dk_qk+dk_kk: max_diff={d5.max():.6f}")

# Show values at first mismatch
mask = d3 > 0.008
if mask.any():
    idx = mask.nonzero()[0]
    pos = tuple(idx.tolist())
    print(f"\nFirst mismatch at {pos}:")
    print(f"  CUDA both:  {dk_both_c[pos].item():.6f}")
    print(f"  FLA both:   {dk_both_f[pos].item():.6f}")
    print(f"  CUDA qk:    {dk_qk_c[pos].item():.6f}")
    print(f"  FLA qk:     {dk_qk_f[pos].item():.6f}")
    print(f"  CUDA kk:    {dk_kk_c[pos].item():.6f}")
    print(f"  FLA kk:     {dk_kk_f[pos].item():.6f}")
    print(f"  CUDA qk+kk: {dk_sum_c[pos].item():.6f}")
    print(f"  FLA qk+kk:  {dk_sum_f[pos].item():.6f}")
