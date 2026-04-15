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
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as fla_chunk_kda_bwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices

from cula.cudac import chunk_kda_bwd_intra_sm90

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


def prepare_bwd_intra_inputs(batch_size, T, H, D, device, cu_seqlens=None, chunk_size=BT, seed=SEED):
    """Prepare inputs for backward intra benchmark.

    FLA uses [1, total_len, H, K] layout (flattened for cu_seqlens);
    SM90 uses [total_q_len, H, K] packed layout.
    Returns both formats.
    """
    from einops import rearrange

    dtype = torch.bfloat16
    set_seed(seed)

    total_q_len = batch_size * T

    # Generate in [B, T, ...] then flatten to [1, B*T, ...] for cu_seqlens compatibility
    q_raw = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    k_raw = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    g_raw = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    beta_raw = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid()

    # Flatten to B=1 for gate preprocessing (required by kda_gate_chunk_cumsum)
    if batch_size != 1:
        q_flat, k_flat, g_flat, beta_flat = map(
            lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q_raw, k_raw, g_raw, beta_raw)
        )
    else:
        q_flat, k_flat, g_flat, beta_flat = q_raw, k_raw, g_raw, beta_raw

    # Gate preprocessing
    A_log = torch.randn(H, dtype=torch.float, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device)

    chunk_indices_fla = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    g_fla = kda_gate_chunk_cumsum(
        g=g_flat,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_fla,
        lower_bound=-5.0,
    )

    # FLA layout: [1, total_q_len, H, ...]
    q_fla = q_flat
    k_fla = k_flat
    beta_fla = beta_flat

    # Backward-specific inputs in FLA layout [1, total_q_len, H, ...]
    dAqk_fla = torch.randn(1, total_q_len, H, BT, dtype=torch.float, device=device) * 0.01
    dAkk_fla = torch.randn(1, total_q_len, H, BT, dtype=torch.float, device=device) * 0.01
    dq_fla = torch.randn(1, total_q_len, H, D, dtype=torch.float, device=device) * 0.01
    dk_fla = torch.randn(1, total_q_len, H, D, dtype=torch.float, device=device) * 0.01
    db_fla = torch.randn(1, total_q_len, H, dtype=torch.float, device=device) * 0.01
    dg_fla = torch.randn(1, total_q_len, H, D, dtype=torch.float, device=device) * 0.01

    # --- SM90 layout: [total_q_len, H, K] packed ---
    q_sm90 = q_fla.reshape(total_q_len, H, D).contiguous()
    k_sm90 = k_fla.reshape(total_q_len, H, D).contiguous()
    g_sm90 = g_fla.reshape(total_q_len, H, D).contiguous()
    beta_sm90 = beta_fla.reshape(total_q_len, H).contiguous()
    dAqk_sm90 = dAqk_fla.reshape(total_q_len, H, BT).contiguous()
    dAkk_sm90 = dAkk_fla.reshape(total_q_len, H, BT).contiguous()
    dq_sm90 = dq_fla.reshape(total_q_len, H, D).contiguous()
    dk_sm90 = dk_fla.reshape(total_q_len, H, D).contiguous()
    db_sm90 = db_fla.reshape(total_q_len, H).contiguous()
    dg_sm90 = dg_fla.reshape(total_q_len, H, D).contiguous()

    # SM90 cu_seqlens and chunk_indices
    if cu_seqlens is None:
        cu_seqlens_sm90 = torch.arange(0, (batch_size + 1) * T, T, dtype=torch.int32, device=device)
    else:
        cu_seqlens_sm90 = cu_seqlens.to(torch.int32)

    # Build SM90 chunk_indices: [num_chunks, 2] → flattened [num_chunks * 2]
    chunk_indices_list = []
    cu = cu_seqlens_sm90.cpu().tolist()
    for b_idx in range(len(cu) - 1):
        seq_start = cu[b_idx]
        seq_len = cu[b_idx + 1] - seq_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for c_idx in range(num_chunks):
            chunk_indices_list.extend([b_idx, c_idx])
    chunk_indices_sm90 = torch.tensor(chunk_indices_list, dtype=torch.int32, device=device)

    fla_inputs = dict(
        q=q_fla,
        k=k_fla,
        g=g_fla,
        beta=beta_fla,
        dAqk=dAqk_fla,
        dAkk=dAkk_fla,
        dq=dq_fla,
        dk=dk_fla,
        db=db_fla,
        dg=dg_fla,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_fla,
        chunk_size=chunk_size,
    )

    sm90_inputs = dict(
        q=q_sm90,
        k=k_sm90,
        g=g_sm90,
        beta=beta_sm90,
        dAqk=dAqk_sm90,
        dAkk=dAkk_sm90,
        dq=dq_sm90,
        dk=dk_sm90,
        db=db_sm90,
        dg=dg_sm90,
        cu_seqlens=cu_seqlens_sm90,
        chunk_indices=chunk_indices_sm90,
        chunk_size=chunk_size,
        total_q_len=total_q_len,
    )

    return fla_inputs, sm90_inputs


def run_fla_bwd_intra(fla_inputs):
    return fla_chunk_kda_bwd_intra(
        q=fla_inputs["q"],
        k=fla_inputs["k"],
        g=fla_inputs["g"],
        beta=fla_inputs["beta"],
        dAqk=fla_inputs["dAqk"],
        dAkk=fla_inputs["dAkk"],
        dq=fla_inputs["dq"],
        dk=fla_inputs["dk"],
        db=fla_inputs["db"],
        dg=fla_inputs["dg"],
        cu_seqlens=fla_inputs["cu_seqlens"],
        chunk_indices=fla_inputs["chunk_indices"],
        chunk_size=fla_inputs["chunk_size"],
        safe_gate=True,
    )


def run_sm90_bwd_intra(sm90_inputs):
    si = sm90_inputs
    total_q_len = si["total_q_len"]
    H_val = si["q"].size(1)
    D_val = si["q"].size(2)

    dq_out = torch.zeros(total_q_len, H_val, D_val, dtype=torch.float32, device=si["q"].device)
    dk_out = torch.zeros(total_q_len, H_val, D_val, dtype=torch.float32, device=si["q"].device)
    db_out = torch.zeros(total_q_len, H_val, dtype=torch.float32, device=si["q"].device)
    dg_out = torch.zeros(total_q_len, H_val, D_val, dtype=torch.float32, device=si["q"].device)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=si["q"].device)

    chunk_kda_bwd_intra_sm90(
        si["q"],
        si["k"],
        si["g"],
        si["beta"],
        si["dAqk"],
        si["dAkk"],
        si["dq"],
        si["dk"],
        si["db"],
        si["dg"],
        si["cu_seqlens"],
        si["chunk_indices"],
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        si["chunk_size"],
    )
    return dq_out, dk_out, db_out, dg_out


# ==============================================================================
# Uniform seqlen benchmark
# ==============================================================================
def benchmark_bwd_intra_uniform():
    device = torch.device("cuda")
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 90)
    print(f"  Uniform-Length BwdIntra Benchmark: SM90 vs FLA Triton  B={B} H={H} D={D}")
    print("=" * 90)
    print(
        f"{'B':>4} {'T':>7} | {'RMSE_dq':>10} {'RMSE_dk':>10} {'RMSE_db':>10} | {'FLA(ms)':>9} {'SM90(ms)':>9} {'Speedup':>8}"
    )
    print("-" * 90)

    for T in T_vals:
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        fla_inputs, sm90_inputs = prepare_bwd_intra_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)

        # Accuracy: run once and compare
        dq_fla, dk_fla, db_fla, dg_fla = run_fla_bwd_intra(fla_inputs)
        dq_sm90, dk_sm90, db_sm90, dg_sm90 = run_sm90_bwd_intra(sm90_inputs)

        total_q_len = B * T
        # FLA outputs [1, total_q_len, H, K], reshape to [total_q_len, H, K] for comparison
        dq_fla_flat = dq_fla.reshape(total_q_len, H, D)
        dk_fla_flat = dk_fla.reshape(total_q_len, H, D)
        db_fla_flat = db_fla.reshape(total_q_len, H)

        rmse_dq, _, _ = accuracy_stats(dq_sm90, dq_fla_flat)
        rmse_dk, _, _ = accuracy_stats(dk_sm90, dk_fla_flat)
        rmse_db, _, _ = accuracy_stats(db_sm90, db_fla_flat)

        # Performance
        ms_fla = triton.testing.do_bench(lambda: run_fla_bwd_intra(fla_inputs))
        ms_sm90 = triton.testing.do_bench(lambda: run_sm90_bwd_intra(sm90_inputs))
        speedup = ms_fla / ms_sm90 if ms_sm90 > 0 else float("inf")

        print(
            f"{B:>4} {T:>7} | {rmse_dq:>10.6f} {rmse_dk:>10.6f} {rmse_db:>10.6f} | {ms_fla:>9.4f} {ms_sm90:>9.4f} {speedup:>7.2f}x"
        )

    print("-" * 90)


# ==============================================================================
# Varlen benchmark
# ==============================================================================
def benchmark_bwd_intra_varlen():
    device = torch.device("cuda")
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 100)
    print(f"  Varlen BwdIntra Benchmark: SM90 vs FLA Triton  NUM_SEQS={NUM_SEQS} H={H} D={D}")
    print("=" * 100)
    print(
        f"{'total_len':>10} | {'RMSE_dq':>10} {'RMSE_dk':>10} {'RMSE_db':>10} | {'FLA(ms)':>9} {'SM90(ms)':>9} {'Speedup':>8}"
    )
    print("-" * 100)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        fla_inputs, sm90_inputs = prepare_bwd_intra_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens)

        # Accuracy
        dq_fla, dk_fla, db_fla, dg_fla = run_fla_bwd_intra(fla_inputs)
        dq_sm90, dk_sm90, db_sm90, dg_sm90 = run_sm90_bwd_intra(sm90_inputs)

        dq_fla_flat = dq_fla.reshape(T, H, D)
        dk_fla_flat = dk_fla.reshape(T, H, D)
        db_fla_flat = db_fla.reshape(T, H)

        rmse_dq, _, _ = accuracy_stats(dq_sm90, dq_fla_flat)
        rmse_dk, _, _ = accuracy_stats(dk_sm90, dk_fla_flat)
        rmse_db, _, _ = accuracy_stats(db_sm90, db_fla_flat)

        # Performance
        ms_fla = triton.testing.do_bench(lambda: run_fla_bwd_intra(fla_inputs))
        ms_sm90 = triton.testing.do_bench(lambda: run_sm90_bwd_intra(sm90_inputs))
        speedup = ms_fla / ms_sm90 if ms_sm90 > 0 else float("inf")

        print(
            f"{total_len:>10} | {rmse_dq:>10.6f} {rmse_dk:>10.6f} {rmse_db:>10.6f} | {ms_fla:>9.4f} {ms_sm90:>9.4f} {speedup:>7.2f}x"
        )

    print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bench_kda_bwd_chunk_intra: SM90 CUDA vs FLA Triton for chunk_kda_bwd_intra")
    parser.parse_args()

    benchmark_bwd_intra_uniform()
    benchmark_bwd_intra_varlen()
