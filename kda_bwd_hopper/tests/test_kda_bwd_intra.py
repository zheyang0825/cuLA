import time
import random
import os
import sys

import torch
from kda.interface import kda_bwd_intra

# Add parent for FLA imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.join(_REPO_ROOT, "kda-backward-internal-main", "third_party", "flash-linear-attention"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "kda-backward-internal-main"))

from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close

torch.backends.cuda.matmul.allow_tf32 = True

from dataclasses import dataclass

@dataclass
class KDAParams:
    B: int
    T: int
    H: int
    K: int
    q: torch.Tensor
    k: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    dAqk: torch.Tensor = None
    dAkk: torch.Tensor = None
    dq: torch.Tensor = None
    dk: torch.Tensor = None
    db: torch.Tensor = None
    dg: torch.Tensor = None
    cu_seqlens: torch.Tensor = None
    chunk_indices: torch.Tensor = None
    chunk_size: int = None

def fla_chunk_kda_bwd_intra(params: KDAParams, safe_gate: bool = False):
    return chunk_kda_bwd_intra(
        params.q, params.k, params.g, params.beta,
        params.dAqk, params.dAkk, params.dq, params.dk, params.db, params.dg,
        params.cu_seqlens, params.chunk_indices, params.chunk_size, safe_gate
    )

def kda_bwd_intra_cuda(params: KDAParams):
    q = params.q
    k = params.k
    g = params.g
    beta = params.beta
    dAqk = params.dAqk
    dAkk = params.dAkk
    dq = params.dq
    dk = params.dk
    db = params.db
    dg = params.dg
    cu_seqlens = params.cu_seqlens
    chunk_indices = params.chunk_indices
    chunk_size = params.chunk_size
    dq_out = torch.empty_like(dq, dtype=torch.bfloat16)
    dk_out = torch.empty_like(dk, dtype=torch.bfloat16)
    db_out = torch.empty_like(db, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    return kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, dq_out, dk_out, db_out, dg_out, chunk_size, tile_counter)

def generate_data(seed=42, B=10, T=800, H=96, K=128, varlen=True):
    torch.manual_seed(seed)
    random.seed(seed)
    BT = 64
    dev = 'cuda'
    cu_seqlens = torch.zeros(B + 1, device=dev, dtype=torch.int32)
    TILE_NUM = 0
    if varlen:
        cu_seqlens[0] = 0
        for i in range(1, B + 1):
            seq_len = min(random.normalvariate(T, T / 2), T*2)
            cu_seqlens[i] = cu_seqlens[i - 1] + int(seq_len)
            TILE_NUM += int((int(seq_len) + 64 - 1) // 64)
    else:
        for i in range(B + 1):
            cu_seqlens[i] = i * T
        TILE_NUM = B * ((T + 64 - 1) // 64)

    chunk_indices = torch.zeros(TILE_NUM * 2, device=dev, dtype=torch.int32)
    acc_num = 0
    for i in range(B):
        seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        current_tile_num = (seq_len + 64 - 1) // 64
        for j in range(current_tile_num):
            chunk_indices[acc_num * 2 + j * 2] = i
            chunk_indices[acc_num * 2 + j * 2 + 1] = j
        acc_num += current_tile_num
    chunk_indices = chunk_indices.reshape(TILE_NUM, 2)

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

    return KDAParams(B=B, T=T, H=H, K=K, q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
                     dq=dq, dk=dk, db=db, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=64)

def test_kda_bwd_intra():
    params = generate_data(seed=42, B=10, T=800, H=96, K=128, varlen=True)

    # Warmup + correctness
    dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)

    # Benchmark CUDA kernel
    for i in range(100):
        dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"CUDA Time taken: {(end_time - start_time) * 1000 / 100:.6f} ms")

    # Determinism check
    dq_out_baseline, dk_out_baseline, db_out_baseline, dg_out_baseline = kda_bwd_intra_cuda(params)
    for i in range(20):
        dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)
        assert torch.equal(dq_out, dq_out_baseline), f"dq not deterministic at iter {i}"
        assert torch.equal(dk_out, dk_out_baseline), f"dk not deterministic at iter {i}"
        assert torch.equal(db_out, db_out_baseline), f"db not deterministic at iter {i}"
        assert torch.equal(dg_out, dg_out_baseline), f"dg not deterministic at iter {i}"

    # FLA baseline
    for i in range(10):
        dq2, dk2, db2, dg2 = fla_chunk_kda_bwd_intra(params, True)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        dq2, dk2, db2, dg2 = fla_chunk_kda_bwd_intra(params, True)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"FLA Time taken: {(end_time - start_time) * 1000 / 100:.6f} ms")

    assert_close("dq", dq2, dq_out, 0.008)
    assert_close("dk", dk2, dk_out, 0.008)
    assert_close("db", db2, db_out, 0.02)
    assert_close("dg", dg2, dg_out.float(), 0.02)
    print("All tests passed!")

if __name__ == "__main__":
    test_kda_bwd_intra()
