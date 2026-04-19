"""
Unit test: verify dK, db, dg computation of kda_bwd_intra_kernel_sm90.

Follows FLA chunk_intra.py algorithm:
  Phase 1: dk2 = sum_{j<=i_i} mask(dAkk[i_i,j]) @ KG, then db=sum(dk2*k), dk2*=beta
  Phase 2: dkt from transposed dA tiles
  Epilogue: dk_out = bf16(dk2 + dk_upstream + dkt)
            dg_out = q * dq2_intra + (dk2 - dkt) * k + dg_upstream
"""
import random
import math
import torch
import cula.cudac as C


def generate_data(seed=42, B=2, T=128, H=2, K=128, varlen=False):
    torch.manual_seed(seed)
    random.seed(seed)
    BT = 64

    cu_seqlens = torch.zeros(B + 1, device="cuda", dtype=torch.int32)
    TILE_NUM = 0
    if varlen:
        for i in range(1, B + 1):
            seq_len = max(int(random.normalvariate(T, T / 2)), BT)
            cu_seqlens[i] = cu_seqlens[i - 1] + seq_len
            TILE_NUM += (seq_len + BT - 1) // BT
    else:
        for i in range(B + 1):
            cu_seqlens[i] = i * T
        TILE_NUM = B * ((T + BT - 1) // BT)

    chunk_indices = torch.zeros(TILE_NUM, 2, device="cuda", dtype=torch.int32)
    acc = 0
    for i in range(B):
        seq_len = int(cu_seqlens[i + 1] - cu_seqlens[i])
        ntiles = (seq_len + BT - 1) // BT
        for j in range(ntiles):
            chunk_indices[acc + j, 0] = i
            chunk_indices[acc + j, 1] = j
        acc += ntiles

    total_len = int(cu_seqlens[-1])
    NK = K // 32

    q = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32) / 10
    beta = torch.randn(1, total_len, H, device="cuda", dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dAkk = torch.randn(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dq = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32)
    dk = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32)
    db = torch.randn(1, total_len, H, device="cuda", dtype=torch.float32)
    dg = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.float32)

    dq_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    dk_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    db_out = torch.zeros(NK, 1, total_len, H, device="cuda", dtype=torch.float32)
    dg_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)

    return dict(
        q=q, k=k, g=g, beta=beta,
        dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        dq_out=dq_out, dk_out=dk_out, db_out=db_out, dg_out=dg_out,
        chunk_size=BT, total_len=total_len, H=H, K=K, NK=NK, B=B,
    )


def compute_expected(data):
    """Reference dK/db/dg computation following FLA chunk_intra.py."""
    BT = data["chunk_size"]
    BC = 16
    BK = 32
    NC = BT // BC
    NK = data["NK"]
    H = data["H"]
    K = data["K"]
    B = data["B"]
    cu_seqlens = data["cu_seqlens"]

    q = data["q"].float()
    k = data["k"].float()
    g = data["g"]
    beta_t = data["beta"].float()
    dAqk = data["dAqk"]
    dAkk = data["dAkk"]
    dq_upstream = data["dq"]
    dk_upstream = data["dk"]
    dg_upstream = data["dg"]

    exp_dk = torch.zeros_like(data["dk_out"], dtype=torch.float32)
    exp_db = torch.zeros_like(data["db_out"])  # [NK, 1, total_len, H]
    exp_dg = torch.zeros_like(data["dg_out"])

    for i_n in range(B):
        bos = int(cu_seqlens[i_n])
        eos = int(cu_seqlens[i_n + 1])
        T_seq = eos - bos

        for i_h in range(H):
            for i_t in range((T_seq + BT - 1) // BT):
                for i_i in range(NC):
                    i_ti = i_t * BT + i_i * BC
                    if i_ti >= T_seq:
                        continue
                    end_r = min(i_ti + BC, T_seq)
                    nr = end_r - i_ti

                    NC_eff = min(NC, math.ceil((T_seq - i_t * BT) / BC))

                    for i_k in range(NK):
                        ks = i_k * BK
                        ke = ks + BK

                        G_i = g[0, bos+i_ti:bos+end_r, i_h, ks:ke]  # [nr, BK]
                        Q_i = q[0, bos+i_ti:bos+end_r, i_h, ks:ke]  # [nr, BK]
                        K_i = k[0, bos+i_ti:bos+end_r, i_h, ks:ke]  # [nr, BK]
                        B_i = beta_t[0, bos+i_ti:bos+end_r, i_h]    # [nr]

                        # ── Phase 1: dk2 (same loop as dq2) ──
                        dq2_tile = torch.zeros(nr, BK, device="cuda", dtype=torch.float32)
                        dk2_tile = torch.zeros(nr, BK, device="cuda", dtype=torch.float32)

                        # Off-diagonal (j < i_i)
                        if i_i > 0:
                            gn = g[0, bos+i_ti, i_h, ks:ke].unsqueeze(0)  # [1, BK]
                            for i_j in range(i_i):
                                j_ti = i_t * BT + i_j * BC
                                end_c = min(j_ti + BC, T_seq)
                                nc = end_c - j_ti
                                K_j = k[0, bos+j_ti:bos+end_c, i_h, ks:ke].float()
                                G_j = g[0, bos+j_ti:bos+end_c, i_h, ks:ke]
                                KG = K_j * torch.exp2(gn - G_j)

                                dA_qk = torch.zeros(nr, nc, device="cuda")
                                dA_kk = torch.zeros(nr, nc, device="cuda")
                                for r in range(nr):
                                    for c in range(nc):
                                        dA_qk[r, c] = dAqk[0, bos+i_ti+r, i_h, i_j*BC+c]
                                        dA_kk[r, c] = dAkk[0, bos+i_ti+r, i_h, i_j*BC+c]
                                dq2_tile += dA_qk @ KG
                                dk2_tile += dA_kk @ KG

                            scale = torch.exp2(G_i - gn)
                            dq2_tile *= scale
                            dk2_tile *= scale

                        # Diagonal (j == i_i, SAFE_GATE)
                        mid = min(BC // 2, T_seq - i_ti - 1)
                        gn_diag = g[0, bos+i_ti+mid, i_h, ks:ke].unsqueeze(0)  # [1, BK]

                        dA_qk_diag = torch.zeros(nr, nr, device="cuda")
                        dA_kk_diag = torch.zeros(nr, nr, device="cuda")
                        for r in range(nr):
                            for c in range(nr):
                                dA_qk_diag[r, c] = dAqk[0, bos+i_ti+r, i_h, i_i*BC+c]
                                dA_kk_diag[r, c] = dAkk[0, bos+i_ti+r, i_h, i_i*BC+c]
                        # Lower-tri mask
                        for r in range(nr):
                            for c in range(nr):
                                if r < c:
                                    dA_qk_diag[r, c] = 0.0
                                    dA_kk_diag[r, c] = 0.0

                        g_diff = G_i[:nr] - gn_diag
                        k_exp = K_i[:nr] * torch.exp2(-g_diff)
                        exp_scale = torch.exp2(g_diff)

                        dq2_tile += (dA_qk_diag[:nr, :nr] @ k_exp) * exp_scale
                        dk2_tile += (dA_kk_diag[:nr, :nr] @ k_exp) * exp_scale

                        # ── Intermediate ──
                        db_val = (dk2_tile * K_i[:nr]).sum(dim=1)  # [nr]
                        dk2_tile *= B_i[:nr, None]  # dk2 *= beta

                        # dg2_part = q * dq2_intra
                        dg2_part = Q_i[:nr] * dq2_tile

                        # ── Phase 2: dkt ──
                        dkt_tile = torch.zeros(nr, BK, device="cuda", dtype=torch.float32)

                        # Off-diagonal (j > i_i)
                        if i_i < NC_eff - 1:
                            last_local = min(BC, T_seq - i_ti) - 1
                            gn_off = g[0, bos+i_ti+last_local, i_h, ks:ke].unsqueeze(0)

                            for i_j in range(i_i + 1, NC_eff):
                                j_ti = i_t * BT + i_j * BC
                                end_c = min(j_ti + BC, T_seq)
                                nc = end_c - j_ti

                                # Transposed dA: sDA(r,c) = dA[j_row+c, i_col+r]
                                dAt_qk = torch.zeros(nr, nc, device="cuda")
                                dAt_kk = torch.zeros(nr, nc, device="cuda")
                                for r in range(nr):
                                    for c in range(nc):
                                        dAt_qk[r, c] = dAqk[0, bos+j_ti+c, i_h, i_i*BC+r]
                                        dAt_kk[r, c] = dAkk[0, bos+j_ti+c, i_h, i_i*BC+r]

                                Q_j = q[0, bos+j_ti:bos+end_c, i_h, ks:ke].float()
                                K_j = k[0, bos+j_ti:bos+end_c, i_h, ks:ke].float()
                                G_j = g[0, bos+j_ti:bos+end_c, i_h, ks:ke]
                                B_j = beta_t[0, bos+j_ti:bos+end_c, i_h]

                                gating = torch.exp2(G_j - gn_off)
                                QG = Q_j * gating
                                KBG = K_j * B_j[:, None] * gating

                                dkt_tile += dAt_qk[:nr, :nc] @ QG[:nc]
                                dkt_tile += dAt_kk[:nr, :nc] @ KBG[:nc]

                            dkt_tile *= torch.exp2(gn_off - G_i[:nr])

                        # Diagonal (j == i_i, upper-tri)
                        gn_diag2 = gn_diag  # same anchor as Phase 1 diagonal

                        dAt_qk_diag = torch.zeros(nr, nr, device="cuda")
                        dAt_kk_diag = torch.zeros(nr, nr, device="cuda")
                        for r in range(nr):
                            for c in range(nr):
                                dAt_qk_diag[r, c] = dAqk[0, bos+i_ti+c, i_h, i_i*BC+r]
                                dAt_kk_diag[r, c] = dAkk[0, bos+i_ti+c, i_h, i_i*BC+r]
                        # Upper-tri mask (r <= c)
                        for r in range(nr):
                            for c in range(nr):
                                if r > c:
                                    dAt_qk_diag[r, c] = 0.0
                                    dAt_kk_diag[r, c] = 0.0

                        g_diff2 = G_i[:nr] - gn_diag2
                        q_exp = Q_i[:nr] * torch.exp2(g_diff2)
                        kb_exp = K_i[:nr] * B_i[:nr, None] * torch.exp2(g_diff2)
                        scale2 = torch.exp2(-g_diff2)

                        dkt_tile += (dAt_qk_diag[:nr, :nr] @ q_exp) * scale2
                        dkt_tile += (dAt_kk_diag[:nr, :nr] @ kb_exp) * scale2

                        # ── Epilogue ──
                        dg_val = dg2_part + (dk2_tile - dkt_tile) * K_i[:nr] + dg_upstream[0, bos+i_ti:bos+end_r, i_h, ks:ke]
                        dk_val = dk2_tile + dk_upstream[0, bos+i_ti:bos+end_r, i_h, ks:ke] + dkt_tile

                        exp_dk[0, bos+i_ti:bos+end_r, i_h, ks:ke] = dk_val
                        exp_dg[0, bos+i_ti:bos+end_r, i_h, ks:ke] = dg_val
                        exp_db[i_k, 0, bos+i_ti:bos+end_r, i_h] = db_val

    return exp_dk.to(torch.bfloat16), exp_db, exp_dg


def run_and_check(label, data,
                  dk_atol=0.3, dk_rtol=0.02,   # bf16 output + TF32 MMA: ULP@16=0.125, @32=0.250
                  db_atol=0.25, db_rtol=0.02,   # fp32 output, TF32 accumulation in dk2
                  dg_atol=0.12, dg_rtol=0.02):  # fp32 output, TF32 accumulation
    C.chunk_kda_bwd_intra_cuda(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["dq_out"], data["dk_out"], data["db_out"], data["dg_out"],
        data["chunk_size"],
    )
    torch.cuda.synchronize()

    exp_dk, exp_db, exp_dg = compute_expected(data)

    ok_all = True
    BT = data["chunk_size"]
    BC = 16
    BK = 32
    cu = data["cu_seqlens"]

    # Check dk_out
    actual_dk = data["dk_out"]
    max_diff = (exp_dk.float() - actual_dk.float()).abs().max().item()
    mean_diff = (exp_dk.float() - actual_dk.float()).abs().mean().item()
    close_dk = torch.allclose(exp_dk.float(), actual_dk.float(), atol=dk_atol, rtol=dk_rtol)
    tag = "PASS" if close_dk else "FAIL"
    print(f"  [{tag}] dk_out  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    if not close_dk:
        ok_all = False
        diff = (exp_dk.float() - actual_dk.float()).abs()
        for i_n in range(data["B"]):
            bos = int(cu[i_n]); T_seq = int(cu[i_n+1]) - bos
            for i_h in range(data["H"]):
                for i_t in range((T_seq + BT - 1) // BT):
                    for i_i in range(BT // BC):
                        i_ti = i_t * BT + i_i * BC
                        if i_ti >= T_seq: continue
                        end_r = min(i_ti + BC, T_seq)
                        for i_k in range(data["NK"]):
                            ks = i_k * BK
                            td = diff[0, bos+i_ti:bos+end_r, i_h, ks:ks+BK].max().item()
                            if td > dk_atol:
                                print(f"    tile n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}: max_diff={td:.6f}")

    # Check db_out
    actual_db = data["db_out"]
    max_diff_db = (exp_db - actual_db).abs().max().item()
    mean_diff_db = (exp_db - actual_db).abs().mean().item()
    close_db = torch.allclose(exp_db, actual_db, atol=db_atol, rtol=db_rtol)
    tag = "PASS" if close_db else "FAIL"
    print(f"  [{tag}] db_out  max_diff={max_diff_db:.6f}  mean_diff={mean_diff_db:.6f}")
    if not close_db:
        ok_all = False

    # Check dg_out
    actual_dg = data["dg_out"]
    max_diff_dg = (exp_dg - actual_dg).abs().max().item()
    mean_diff_dg = (exp_dg - actual_dg).abs().mean().item()
    close_dg = torch.allclose(exp_dg, actual_dg, atol=dg_atol, rtol=dg_rtol)
    tag = "PASS" if close_dg else "FAIL"
    print(f"  [{tag}] dg_out  max_diff={max_diff_dg:.6f}  mean_diff={mean_diff_dg:.6f}")
    if not close_dg:
        ok_all = False
        diff = (exp_dg - actual_dg).abs()
        for i_n in range(data["B"]):
            bos = int(cu[i_n]); T_seq = int(cu[i_n+1]) - bos
            for i_h in range(data["H"]):
                for i_t in range((T_seq + BT - 1) // BT):
                    for i_i in range(BT // BC):
                        i_ti = i_t * BT + i_i * BC
                        if i_ti >= T_seq: continue
                        end_r = min(i_ti + BC, T_seq)
                        for i_k in range(data["NK"]):
                            ks = i_k * BK
                            td = diff[0, bos+i_ti:bos+end_r, i_h, ks:ks+BK].max().item()
                            if td > dg_atol:
                                print(f"    tile n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}: max_diff={td:.6f}")

    status = "PASS" if ok_all else "FAIL"
    print(f"[{status}] {label}\n")
    return ok_all


if __name__ == "__main__":
    ok = True
    print("=== Fixed-length test ===")
    d = generate_data(seed=42, B=2, T=128, H=2, K=128, varlen=False)
    ok &= run_and_check("fixed B=2 T=128 H=2 K=128", d)

    print("=== Variable-length test ===")
    d = generate_data(seed=123, B=4, T=96, H=2, K=128, varlen=True)
    ok &= run_and_check("varlen B=4 T=96 H=2 K=128", d)

    print("=== Minimal test (1 chunk) ===")
    d = generate_data(seed=7, B=1, T=64, H=1, K=128, varlen=False)
    ok &= run_and_check("minimal B=1 T=64 H=1 K=128", d)

    exit(0 if ok else 1)
