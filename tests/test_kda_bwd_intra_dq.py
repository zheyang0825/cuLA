"""
Unit test: verify dQ computation of kda_bwd_intra_kernel_sm90.

dQ_intra[i_i, :, k] = sum_{j<=i_i} mask(dAqk[i_i,j]) @ (K[j,:,k] * exp2(-G[j,:,k])) * exp2(G[i_i,:,k])
dq_out = bf16(dQ_intra + dq_upstream)

where mask is lower-triangular for j==i_i, all-ones for j<i_i.
"""
import random
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


def compute_expected_dq(data):
    """Reference dQ computation in PyTorch."""
    BT = data["chunk_size"]
    BC = 16
    BK = 32
    NC = BT // BC
    NK = data["NK"]
    H = data["H"]
    K = data["K"]
    B = data["B"]
    cu_seqlens = data["cu_seqlens"]

    q = data["q"]
    k = data["k"]
    g = data["g"]
    dAqk = data["dAqk"]
    dq_upstream = data["dq"]

    # Output: same shape as dq_out
    expected = torch.zeros_like(data["dq_out"], dtype=torch.float32)

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
                    nr = end_r - i_ti  # actual rows in this tile

                    for i_k in range(NK):
                        ks = i_k * BK
                        ke = ks + BK

                        # G values for this tile's rows
                        G_i = g[0, bos + i_ti : bos + end_r, i_h, ks:ke]  # [nr, BK]

                        dq_tile = torch.zeros(nr, BK, device="cuda", dtype=torch.float32)

                        for i_j in range(i_i + 1):
                            j_ti = i_t * BT + i_j * BC
                            end_c = min(j_ti + BC, T_seq)
                            nc = end_c - j_ti

                            # dAqk block: rows from tile i_i, cols from tile i_j
                            # dAqk shape: [1, total_len, H, BT]
                            # row index: bos + i_ti + r, col within BT: i_j * BC + c
                            dA_block = torch.zeros(nr, nc, device="cuda", dtype=torch.float32)
                            for r in range(nr):
                                for c in range(nc):
                                    dA_block[r, c] = dAqk[0, bos + i_ti + r, i_h, i_j * BC + c]

                            # Apply mask
                            if i_j == i_i:
                                # Lower-triangular: keep r >= c
                                for r in range(nr):
                                    for c in range(nc):
                                        if r < c:
                                            dA_block[r, c] = 0.0

                            # K and G for tile j
                            K_j = k[0, bos + j_ti : bos + end_c, i_h, ks:ke].float()  # [nc, BK]
                            G_j = g[0, bos + j_ti : bos + end_c, i_h, ks:ke]          # [nc, BK]

                            # K_j_scaled = K_j * exp2(-G_j)
                            K_j_scaled = K_j * torch.exp2(-G_j)

                            # dq_tile += dA_block @ K_j_scaled * exp2(G_i)
                            # dA_block: [nr, nc], K_j_scaled: [nc, BK] → [nr, BK]
                            tmp = dA_block[:nr, :nc] @ K_j_scaled[:nc, :]
                            dq_tile[:nr, :] += tmp * torch.exp2(G_i[:nr, :])

                        # Store
                        expected[0, bos + i_ti : bos + end_r, i_h, ks:ke] = dq_tile

    # Add upstream and convert to bf16
    expected_with_upstream = expected + dq_upstream
    return expected_with_upstream.to(torch.bfloat16)


def run_and_check(label, data, atol=1e-2, rtol=1e-2):
    C.chunk_kda_bwd_intra_cuda(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["dq_out"], data["dk_out"], data["db_out"], data["dg_out"],
        data["chunk_size"],
    )
    torch.cuda.synchronize()

    expected_dq = compute_expected_dq(data)
    actual_dq = data["dq_out"]

    max_diff = (expected_dq.float() - actual_dq.float()).abs().max().item()
    mean_diff = (expected_dq.float() - actual_dq.float()).abs().mean().item()

    close = torch.allclose(expected_dq.float(), actual_dq.float(), atol=atol, rtol=rtol)

    if close:
        print(f"[PASS] {label}  (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
    else:
        print(f"[FAIL] {label}  (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        # Show first few mismatches
        diff = (expected_dq.float() - actual_dq.float()).abs()
        worst_idx = diff.argmax()
        flat_exp = expected_dq.float().flatten()
        flat_act = actual_dq.float().flatten()
        print(f"  worst: expected={flat_exp[worst_idx].item():.6f}, actual={flat_act[worst_idx].item():.6f}")

        # Check per-tile
        BT = data["chunk_size"]
        BC = 16
        BK = 32
        cu = data["cu_seqlens"]
        for i_n in range(data["B"]):
            bos = int(cu[i_n])
            eos = int(cu[i_n + 1])
            T_seq = eos - bos
            for i_h in range(data["H"]):
                for i_t in range((T_seq + BT - 1) // BT):
                    for i_i in range(BT // BC):
                        i_ti = i_t * BT + i_i * BC
                        if i_ti >= T_seq:
                            continue
                        end_r = min(i_ti + BC, T_seq)
                        for i_k in range(data["NK"]):
                            ks = i_k * BK
                            ke = ks + BK
                            tile_diff = diff[0, bos+i_ti:bos+end_r, i_h, ks:ke].max().item()
                            if tile_diff > atol:
                                print(f"  tile n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}: max_diff={tile_diff:.6f}")

    return close


if __name__ == "__main__":
    ok = True
    print("=== Fixed-length test ===")
    d = generate_data(seed=42, B=2, T=128, H=2, K=128, varlen=False)
    ok &= run_and_check("fixed B=2 T=128 H=2 K=128", d)

    print("\n=== Variable-length test ===")
    d = generate_data(seed=123, B=4, T=96, H=2, K=128, varlen=True)
    ok &= run_and_check("varlen B=4 T=96 H=2 K=128", d)

    print("\n=== Minimal test (1 chunk) ===")
    d = generate_data(seed=7, B=1, T=64, H=1, K=128, varlen=False)
    ok &= run_and_check("minimal B=1 T=64 H=1 K=128", d)

    exit(0 if ok else 1)
