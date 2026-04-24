"""Minimal NCU driver for the SM90 KDA bwd intra kernel."""
import argparse
import os
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "flash-linear-attention"))
os.environ.setdefault("FLA_USE_FAST_OPS", "1")

from cula.cudac import chunk_kda_bwd_intra_sm90  # noqa: E402
from fla.ops.utils.index import prepare_chunk_indices  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=4)
p.add_argument("--T", type=int, default=2048)
p.add_argument("--H", type=int, default=4)
p.add_argument("--D", type=int, default=128)
p.add_argument("--BT", type=int, default=64)
args = p.parse_args()

torch.manual_seed(0)
B, T, H, D, BT = args.B, args.T, args.H, args.D, args.BT
total = B * T
device = "cuda"

q = torch.randn(total, H, D, device=device, dtype=torch.bfloat16)
k = torch.randn(total, H, D, device=device, dtype=torch.bfloat16)
g = torch.randn(total, H, D, device=device, dtype=torch.float32) * 0.1
beta = torch.randn(total, H, device=device, dtype=torch.float32)
dAqk = torch.randn(total, H, BT, device=device, dtype=torch.float32) * 0.01
dAkk = torch.randn(total, H, BT, device=device, dtype=torch.float32) * 0.01
dq = torch.randn(total, H, D, device=device, dtype=torch.float32) * 0.01
dk = torch.randn(total, H, D, device=device, dtype=torch.float32) * 0.01
dg = torch.randn(total, H, D, device=device, dtype=torch.float32) * 0.01
db = torch.randn(total, H, device=device, dtype=torch.float32) * 0.01
dq_out = torch.zeros_like(dq); dk_out = torch.zeros_like(dk)
dg_out = torch.zeros_like(dg); db_out = torch.zeros_like(db)
tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

cu_seqlens = torch.arange(0, B + 1, device=device, dtype=torch.int32) * T
chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

def call():
    chunk_kda_bwd_intra_sm90(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                             cu_seqlens, chunk_indices,
                             dq_out, dk_out, db_out, dg_out, tile_counter, BT)

for _ in range(3):
    call()
torch.cuda.synchronize()
call()
torch.cuda.synchronize()
