"""Benchmark SM90 KDA Backward dqkg kernel vs FLA Triton reference."""
import math, sys, time
import torch
sys.path.insert(0, "third_party/flash-linear-attention")

def make_inputs(B, T, H, K=128, V=128, seed=42):
    torch.manual_seed(seed)
    dev, bf, fp = "cuda", torch.bfloat16, torch.float32
    q     = torch.randn(B*T, H, K, dtype=bf, device=dev)
    k     = torch.randn(B*T, H, K, dtype=bf, device=dev)
    v     = torch.randn(B*T, H, V, dtype=bf, device=dev)
    v_new = torch.randn(B*T, H, V, dtype=bf, device=dev)
    g     = -torch.rand(B*T, H, K, dtype=fp, device=dev) * 0.5
    beta  = torch.rand(B*T, H, dtype=fp, device=dev) * 0.5 + 0.1
    A     = torch.randn(B*T, H, 64, dtype=bf, device=dev) * 0.1
    NT    = (T + 63) // 64
    h     = torch.randn(NT*B, H, K, V, dtype=bf, device=dev) * 0.1
    dh    = torch.randn(NT*B, H, K, V, dtype=bf, device=dev) * 0.1
    do_   = torch.randn(B*T, H, V, dtype=bf, device=dev)
    dv    = torch.randn(B*T, H, V, dtype=bf, device=dev)
    return q, k, v, v_new, g, beta, A, h, do_, dh, dv

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms

configs = [
    (1,  64, 1),
    (1, 128, 2),
    (2, 256, 4),
    (2, 2048, 8),
]

from cula._kda_bwd_dqkg_sm90 import chunk_kda_bwd_dqkg_sm90
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused

print(f"{'Shape':>20}  {'FLA (ms)':>10}  {'cuLA (ms)':>10}  {'Speedup':>8}")
print("-" * 60)
for (B, T, H) in configs:
    K, V = 128, 128
    scale = 1.0 / math.sqrt(K)
    q, k, v, v_new, g, beta, A, h, do_, dh, dv = make_inputs(B, T, H)

    q4    = q.reshape(B,T,H,K);  k4    = k.reshape(B,T,H,K)
    v4    = v.reshape(B,T,H,V);  vn4   = v_new.reshape(B,T,H,V)
    g4    = g.reshape(B,T,H,K);  b4    = beta.reshape(B,T,H)
    A4    = A.reshape(B,T,H,64); do4   = do_.reshape(B,T,H,V)
    dv4   = dv.reshape(B,T,H,V)

    t_fla  = bench(lambda: chunk_kda_bwd_wy_dqkg_fused(
        q=q4,k=k4,v=v4,v_new=vn4,g=g4,beta=b4,A=A4,h=h,do=do4,dh=dh,dv=dv4,
        scale=scale,chunk_size=64,transpose_state_layout=False))
    t_cula = bench(lambda: chunk_kda_bwd_dqkg_sm90(
        q,k,v,v_new,g,beta,A,h,do_,dh,dv,scale,B,T))

    tag = f"B={B},T={T},H={H}"
    print(f"{tag:>20}  {t_fla:>10.3f}  {t_cula:>10.3f}  {t_fla/t_cula:>8.2f}x")
