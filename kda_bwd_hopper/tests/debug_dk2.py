"""Debug dk - test with dAkk and beta."""
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

def compare(name, ref, test, tol=0.008):
    diff = (ref.float() - test.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    error_rate = (diff > tol).float().mean().item() * 100
    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, error_rate={error_rate:.2f}% (tol={tol})")

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
beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
dAqk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)
dAkk = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32)

tests = [
    ("dAqk only, g=0, beta=0",    dict(g_zero=True, beta_zero=True, dAkk_zero=True)),
    ("dAkk only, g=0, beta=0.5",  dict(g_zero=True, beta_val=0.5, dAqk_zero=True)),
    ("both dA, g=0, beta=rand",   dict(g_zero=True)),
    ("dAqk only, g=rand/10, beta=0", dict(beta_zero=True, dAkk_zero=True)),
    ("full (all non-zero)",        dict()),
]

for desc, opts in tests:
    g = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32) if opts.get('g_zero') else torch.randn(1, T, H, K, device=dev, dtype=torch.float32) / 10
    b = torch.zeros(1, T, H, device=dev, dtype=torch.bfloat16) if opts.get('beta_zero') else beta
    if opts.get('beta_val') is not None:
        b = torch.full((1, T, H), opts['beta_val'], device=dev, dtype=torch.bfloat16)
    aqk = torch.zeros_like(dAqk) if opts.get('dAqk_zero') else dAqk
    akk = torch.zeros_like(dAkk) if opts.get('dAkk_zero') else dAkk

    dq_z = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
    dk_z = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)
    db_z = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
    dg_z = torch.zeros(1, T, H, K, device=dev, dtype=torch.float32)

    dq_c, dk_c, db_c, dg_c = run_cuda(q, k, g, b, aqk, akk, dq_z, dk_z, db_z, dg_z, cu_seqlens, chunk_indices, BT)
    dq_f, dk_f, db_f, dg_f = run_fla(q, k, g, b, aqk, akk, dq_z.clone(), dk_z.clone(), db_z.clone(), dg_z.clone(), cu_seqlens, chunk_indices, BT)

    print(f"\n{desc}:")
    compare("dq", dq_f, dq_c)
    compare("dk", dk_f, dk_c)
    compare("db", db_f, db_c, tol=0.02)
    compare("dg", dg_f, dg_c, tol=0.02)
