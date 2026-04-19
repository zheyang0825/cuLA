"""
Debug test: isolate dQ issues by using g=0 (all gate terms become 1).
With g=0, dQ_intra = sum_j mask(dAqk) @ K, no exp2 scaling.
"""
import torch
import cula.cudac as C


def test_zero_gate():
    """Test with g=0: eliminates all gate normalization, tests raw GEMM path."""
    torch.manual_seed(42)
    B, T, H, K = 1, 64, 1, 128
    BT, BC, BK = 64, 16, 32
    NC = BT // BC
    NK = K // BK

    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    chunk_indices = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
    total_len = T

    q = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)  # ZERO
    beta = torch.randn(1, total_len, H, device="cuda", dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dAkk = torch.zeros(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dq = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)  # zero upstream
    dk = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)
    db = torch.zeros(1, total_len, H, device="cuda", dtype=torch.float32)
    dg = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)

    dq_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    dk_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    db_out = torch.zeros(NK, 1, total_len, H, device="cuda", dtype=torch.float32)
    dg_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)

    C.chunk_kda_bwd_intra_cuda(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
        cu_seqlens, chunk_indices,
        dq_out, dk_out, db_out, dg_out, BT,
    )
    torch.cuda.synchronize()

    # Reference: with g=0, exp2(G_i - G_j) = 1 for all i,j
    # dQ_intra[i, k] = sum_j mask(dAqk[i,j]) * K[j, k]
    expected = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)
    for i_i in range(NC):
        i_ti = i_i * BC
        end_r = min(i_ti + BC, T)
        nr = end_r - i_ti
        for i_k in range(NK):
            ks, ke = i_k * BK, (i_k + 1) * BK
            dq_tile = torch.zeros(nr, BK, device="cuda", dtype=torch.float32)
            for i_j in range(i_i + 1):
                j_ti = i_j * BC
                end_c = min(j_ti + BC, T)
                nc = end_c - j_ti
                # dA block
                dA_block = dAqk[0, i_ti:end_r, 0, i_j*BC:i_j*BC+nc].clone()
                # Lower-tri mask for diagonal
                if i_j == i_i:
                    for r in range(nr):
                        for c in range(nc):
                            if r < c:
                                dA_block[r, c] = 0.0
                K_j = k[0, j_ti:end_c, 0, ks:ke].float()
                dq_tile += dA_block @ K_j
            expected[0, i_ti:end_r, 0, ks:ke] = dq_tile

    expected_bf16 = expected.to(torch.bfloat16)
    actual = dq_out

    for i_i in range(NC):
        i_ti = i_i * BC
        end_r = min(i_ti + BC, T)
        for i_k in range(NK):
            ks, ke = i_k * BK, (i_k + 1) * BK
            exp_tile = expected_bf16[0, i_ti:end_r, 0, ks:ke]
            act_tile = actual[0, i_ti:end_r, 0, ks:ke]
            diff = (exp_tile.float() - act_tile.float()).abs()
            max_d = diff.max().item()
            status = "PASS" if max_d < 0.1 else "FAIL"
            print(f"  [{status}] i_i={i_i} i_k={i_k}: max_diff={max_d:.6f}")
            if max_d >= 0.1:
                # Print first few mismatches
                worst = diff.argmax()
                r, c = worst // BK, worst % BK
                print(f"    worst at ({r},{c}): expected={exp_tile[r,c].item():.4f} actual={act_tile[r,c].item():.4f}")


def test_identity_dA():
    """Test with dAqk = identity blocks and g=0: simplest possible GEMM."""
    torch.manual_seed(42)
    B, T, H, K = 1, 64, 1, 128
    BT, BC, BK = 64, 16, 32
    NC = BT // BC
    NK = K // BK

    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    chunk_indices = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
    total_len = T

    q = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)
    beta = torch.zeros(1, total_len, H, device="cuda", dtype=torch.bfloat16)

    # Set dAqk to identity-like: dAqk[row, col] = 1 if row==col (within BT block)
    dAqk = torch.zeros(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    for r in range(T):
        local_r = r % BT
        if local_r < BT:
            dAqk[0, r, 0, local_r] = 1.0

    dAkk = torch.zeros(1, total_len, H, BT, device="cuda", dtype=torch.float32)
    dq = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)
    dk = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)
    db = torch.zeros(1, total_len, H, device="cuda", dtype=torch.float32)
    dg = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)

    dq_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    dk_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.bfloat16)
    db_out = torch.zeros(NK, 1, total_len, H, device="cuda", dtype=torch.float32)
    dg_out = torch.zeros(1, total_len, H, K, device="cuda", dtype=torch.float32)

    C.chunk_kda_bwd_intra_cuda(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
        cu_seqlens, chunk_indices,
        dq_out, dk_out, db_out, dg_out, BT,
    )
    torch.cuda.synchronize()

    # With identity dAqk and lower-tri mask, row i accumulates K[0:i+1, :]
    # dQ[i, k] = sum_{j<=i_within_tile} K[j, k]
    print("\n  Identity dAqk test (diagonal tile i_i=0 only, i_k=0):")
    i_k = 0
    ks, ke = 0, BK
    for row in range(min(8, BC)):  # first 8 rows of first sub-tile
        # Expected: sum of K[0:row+1, ks:ke]
        exp_val = k[0, 0:row+1, 0, ks:ke].float().sum(dim=0)
        act_val = dq_out[0, row, 0, ks:ke].float()
        diff = (exp_val - act_val).abs().max().item()
        print(f"    row {row}: max_diff={diff:.6f}, exp[0]={exp_val[0]:.4f}, act[0]={act_val[0]:.4f}")


if __name__ == "__main__":
    print("=== Zero-gate test ===")
    test_zero_gate()
    print("\n=== Identity dA test ===")
    test_identity_dA()
