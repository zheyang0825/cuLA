#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NCU profiling script for chunk_kda_bwd_intra kernel.

Usage:
    ncu --set full -o bwd_intra_profile python scripts/ncu_profile_bwd_intra.py

Then open bwd_intra_profile.ncu-rep in Nsight Compute GUI.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.utils.index import prepare_chunk_indices

import cula.cudac as C
from benchmarks.utils import SEED, exclusive_cumsum, set_seed

# ── Config ───────────────────────────────────────────────────────────────────
B, T, H, D = 2, 4096, 64, 128
BT = 64
WARMUP = 3
PROFILE_ITERS = 1
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda")
    total_len = B * T
    NK = D // 32

    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    q = torch.randn(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32) / 10
    beta = torch.randn(1, total_len, H, device=device, dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, BT, device=device, dtype=torch.float32)
    dAkk = torch.randn(1, total_len, H, BT, device=device, dtype=torch.float32)
    dq = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32)
    dk = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32)
    db = torch.randn(1, total_len, H, device=device, dtype=torch.float32)
    dg = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32)

    dq_out = torch.empty(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    dk_out = torch.empty(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    db_out = torch.empty(NK, 1, total_len, H, device=device, dtype=torch.float32)
    dg_out = torch.empty(1, total_len, H, D, device=device, dtype=torch.float32)

    def run():
        C.chunk_kda_bwd_intra_cuda(
            q, k, g, beta,
            dAqk, dAkk, dq, dk, db, dg,
            cu_seqlens, chunk_indices,
            dq_out, dk_out, db_out, dg_out,
            BT,
        )

    # Warmup
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    # Profile region
    for _ in range(PROFILE_ITERS):
        run()
    torch.cuda.synchronize()

    print("Done. Open the .ncu-rep file in Nsight Compute GUI.")


if __name__ == "__main__":
    main()
