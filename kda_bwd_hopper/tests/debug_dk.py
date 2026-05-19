"""Debug dk specifically - isolate backward dkt computation."""
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

# TEST: dAkk=0, so dk = dk_in + dkt (from dAqk only)
# With g=0, dkt = dAqk^T @ q (upper triangular mask, then scale by exp2(gn-g)=1)
print("=" * 60)
print("TEST: dk with dAkk=0, g=0 (isolate dkt_q)")
print("=" * 60)

torch.manual_seed(42)
cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
chunk_indices = torch.tensor([[0, 0]], device=dev, dtype=torch.int32)

q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
g = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
beta = torch.zeros(1, T, H, device=dev, dtype=torch.bfloat16)

dAqk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)
dAkk = torch.zeros(1, T, H, BT, device=dev, dtype=torch.float32)

dq = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
dk = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
db = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
dg = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT)
dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, beta, dAqk, dAkk, dq.clone(), dk.clone(), db.clone(), dg.clone(), cu_seqlens, chunk_indices, BT)

diff_dk = (dk_c.float() - dk_f.float()).abs()
print(f"  dk: max={diff_dk.max():.6f} mean={diff_dk.mean():.6f} error_rate={(diff_dk>0.008).float().mean()*100:.2f}%")
print(f"  dk CUDA[:8]: {dk_c[0,0,0,:8].tolist()}")
print(f"  dk FLA [:8]: {dk_f[0,0,0,:8].tolist()}")

# With beta=0 and dAkk=0: dk_out = dk_in + dk2*beta + dkt = 0 + 0 + dkt
# dkt should be upper_tri(dAqk[:16,:16])^T @ q[:16,:] * exp2(gn-g) (with g=0, scale=1)
# = upper_tri(dAqk[:16,:16])^T @ q[:16,:]
dA_block = dAqk[0, :T, 0, :T]  # [16, 16]
upper_mask = torch.triu(torch.ones(T, T, device=dev))
# dkt[i] = sum_{j>=i} dA[j][i] * q[j]  (transposed and upper triangular)
# upper_tri(dA)^T means: for row i in the transposed matrix, sum cols j where j >= i
# Actually: A^T[i][j] = A[j][i]. Upper tri of A^T means i <= j. So A^T[i][j] nonzero when i <= j.
# dkt[i] = sum_{j>=i} A[j][i] * q[j]  (which is sum over column i of upper tri, i.e., A[j][i] for j>=i)
# Wait: in the FLA code:
# dkt_q = upper_tri(dAqk[i:i+16, j:j+16])^T @ qg[j:j+16]
# For diagonal block i==j:
# upper_tri(dAqk[:16,:16])^T means: mask dAqk with upper triangle, then transpose.
# mask: dAqk[r][c] kept if r <= c (upper tri of dAqk itself)
# transpose: result[i][j] = masked_dAqk[j][i] = dAqk[j][i] if j <= i... wait, that's lower triangle.
# Hmm, I need to be careful about what "upper triangular" means here.

# In FLA code (chunk_intra.py): the diagonal backward uses strict_upper=True
# For backward: for j > i (strict upper), A = dAqk[j_sub, i_sub] (transposed)
# Actually, let me just check if FLA matches a PyTorch reference

q_f32 = q[0, :T, 0, :].float()  # [16, K]
dA_upper = dA_block * upper_mask  # upper triangular dAqk
# dkt = dA_upper^T @ q
dkt_ref = dA_upper.T @ q_f32  # [16, K]

# But wait, the diagonal block should use strict upper (row < col) or upper (row <= col)?
# FLA uses strict_upper for off-diagonal backward, but for diagonal it might use row <= col
strict_upper_mask = torch.triu(torch.ones(T, T, device=dev), diagonal=1)
dA_strict = dA_block * strict_upper_mask
dkt_strict = dA_strict.T @ q_f32

print(f"\n  PyTorch ref (upper incl diag)^T @ q: {dkt_ref[0,:8].tolist()}")
print(f"  PyTorch ref (strict upper)^T @ q:     {dkt_strict[0,:8].tolist()}")
print(f"  dk CUDA (=dkt since dk_in=0, beta=0): {dk_c[0,0,0,:8].float().tolist()}")
print(f"  dk FLA:                                {dk_f[0,0,0,:8].float().tolist()}")

# Check which matches
diff1 = (dk_f[0,:,0,:].float() - dkt_ref).abs().max()
diff2 = (dk_f[0,:,0,:].float() - dkt_strict).abs().max()
print(f"\n  FLA vs upper_incl: max_diff={diff1:.6f}")
print(f"  FLA vs strict_upper: max_diff={diff2:.6f}")

# Now check more detailed dk errors per row
print("\n  Per-row dk errors (CUDA vs FLA):")
for r in range(min(8, T)):
    diff = (dk_c[0,r,0,:].float() - dk_f[0,r,0,:].float()).abs()
    print(f"    Row {r}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")

print("\nDone!")
