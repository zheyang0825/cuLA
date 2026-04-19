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

import argparse
import os
import pathlib
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as fla_chunk_kda_bwd_intra

import cula.cudac as C
from benchmarks.utils import SEED, exclusive_cumsum, generate_random_seq_lens, set_seed

# Constant params
B, H, D = 2, 64, 128
BT = 64  # chunk size

# Varlen benchmark params
NUM_SEQS = 8
TOTAL_LEN = 8192
MIN_SEQ_LEN = 63
VARIANCE = 1.0


def accuracy_stats(a, b):
    """Compute RMSE, relative max diff, and mean absolute difference."""
    a, b = a.float(), b.float()
    diff = a - b
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.abs().max().item()
    denom = b.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.abs().mean().item()
    return rmse, rel_max, mean_diff


def prepare_bwd_intra_inputs(total_len, H, D, device, cu_seqlens, chunk_size=BT, seed=SEED):
    """Prepare inputs for chunk_kda_bwd_intra benchmark."""
    import random
    from fla.ops.utils.index import prepare_chunk_indices

    random.seed(seed)
    set_seed(seed)

    NK = D // 32

    q = torch.randn(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32) / 10
    beta = torch.randn(1, total_len, H, device=device, dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, chunk_size, device=device, dtype=torch.float32)
    dAkk = torch.randn(1, total_len, H, chunk_size, device=device, dtype=torch.float32)
    dq = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32)
    dk = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32)
    db = torch.randn(1, total_len, H, device=device, dtype=torch.float32)
    dg = torch.randn(1, total_len, H, D, device=device, dtype=torch.float32)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    # CUDA kernel output buffers
    dq_out = torch.empty(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    dk_out = torch.empty(1, total_len, H, D, device=device, dtype=torch.bfloat16)
    db_out = torch.empty(NK, 1, total_len, H, device=device, dtype=torch.float32)
    dg_out = torch.empty(1, total_len, H, D, device=device, dtype=torch.float32)

    return dict(
        q=q, k=k, g=g, beta=beta,
        dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        dq_out=dq_out, dk_out=dk_out, db_out=db_out, dg_out=dg_out,
        chunk_size=chunk_size, NK=NK,
    )


def run_cula(data):
    """Run cuLA CUDA kernel."""
    C.chunk_kda_bwd_intra_cuda(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["dq_out"], data["dk_out"], data["db_out"], data["dg_out"],
        data["chunk_size"],
    )


def run_fla(data):
    """Run FLA Triton reference."""
    return fla_chunk_kda_bwd_intra(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["chunk_size"], safe_gate=True,
    )


def compare_outputs(data):
    """Run both and compare. Returns (rmse_dq, rmse_dk, rmse_db, rmse_dg)."""
    data["dq_out"].zero_()
    data["dk_out"].zero_()
    data["db_out"].zero_()
    data["dg_out"].zero_()
    run_cula(data)
    torch.cuda.synchronize()

    dq_fla, dk_fla, db_fla, dg_fla = run_fla(data)

    db_cuda = data["db_out"].sum(0).add_(data["db"])

    stats = {}
    for name, ref, tri in [
        ("dq", dq_fla, data["dq_out"]),
        ("dk", dk_fla, data["dk_out"]),
        ("db", db_fla, db_cuda),
        ("dg", dg_fla, data["dg_out"]),
    ]:
        stats[name] = accuracy_stats(ref, tri)

    return stats


# ==============================================================================
# Uniform seqlen benchmark
# ==============================================================================
def benchmark_bwd_intra_uniform():
    device = torch.device("cuda")
    chunk_size = BT
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 100)
    print(f"  Uniform-Length BwdIntra Benchmark: cuLA vs FLA Triton  B={B} H={H} D={D}")
    print("=" * 100)
    print(
        f"{'B':>4} {'T':>7} │ {'dq_rmse':>10} {'dk_rmse':>10} {'db_rmse':>10} {'dg_rmse':>10}"
        f" │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 100)

    for T in T_vals:
        total_len = B * T
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        data = prepare_bwd_intra_inputs(total_len, H, D, device, cu_seqlens, chunk_size)

        # Accuracy
        stats = compare_outputs(data)

        # Performance: FLA
        ms_fla = triton.testing.do_bench(lambda: run_fla(data))

        # Performance: cuLA
        ms_cula = triton.testing.do_bench(lambda: run_cula(data))

        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{B:>4} {T:>7} │ "
            f"{stats['dq'][0]:>10.6f} {stats['dk'][0]:>10.6f} {stats['db'][0]:>10.6f} {stats['dg'][0]:>10.6f}"
            f" │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 100)


# ==============================================================================
# Varlen benchmark
# ==============================================================================
def benchmark_bwd_intra_varlen():
    device = torch.device("cuda")
    chunk_size = BT
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 110)
    print(f"  Varlen BwdIntra Benchmark: cuLA vs FLA Triton  NUM_SEQS={NUM_SEQS} H={H} D={D}")
    print("=" * 110)
    print(
        f"{'total_len':>10} │ {'dq_rmse':>10} {'dk_rmse':>10} {'db_rmse':>10} {'dg_rmse':>10}"
        f" │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 110)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        data = prepare_bwd_intra_inputs(total_len, H, D, device, cu_seqlens, chunk_size)

        # Accuracy
        stats = compare_outputs(data)

        # Performance: FLA
        ms_fla = triton.testing.do_bench(lambda: run_fla(data))

        # Performance: cuLA
        ms_cula = triton.testing.do_bench(lambda: run_cula(data))

        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{total_len:>10} │ "
            f"{stats['dq'][0]:>10.6f} {stats['dk'][0]:>10.6f} {stats['db'][0]:>10.6f} {stats['dg'][0]:>10.6f}"
            f" │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bench_kda_bwd_intra: cuLA vs FLA Triton for chunk_kda_bwd_intra")
    args = parser.parse_args()

    benchmark_bwd_intra_uniform()
    benchmark_bwd_intra_varlen()
