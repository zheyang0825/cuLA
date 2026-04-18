"""
Unit test: verify that kda_bwd_intra_kernel_sm90 loads persistent tiles
(q, k, g, beta) correctly and writes them back unchanged.

Current kernel has computation #if 0 disabled; it only:
  dq_out[tile] = sQ   (bf16 round-trip, exact)
  dk_out[tile] = sK   (bf16 round-trip, exact)
  dg_out[tile] = sG   (fp32, exact)
  db_out[i_k, tile] = s_beta  (bf16→float, exact)
"""
import random
import torch
import cula.cudac as C  # pybind module


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

    # Outputs
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


def run_and_check(label, data):
    C.chunk_kda_bwd_intra_cuda(
        data["q"], data["k"], data["g"], data["beta"],
        data["dAqk"], data["dAkk"], data["dq"], data["dk"], data["db"], data["dg"],
        data["cu_seqlens"], data["chunk_indices"],
        data["dq_out"], data["dk_out"], data["db_out"], data["dg_out"],
        data["chunk_size"],
    )
    torch.cuda.synchronize()

    q = data["q"]
    k = data["k"]
    g = data["g"]
    beta = data["beta"]
    cu_seqlens = data["cu_seqlens"]

    BT = data["chunk_size"]
    BC = 16
    NK = data["NK"]
    NC = BT // BC
    H = data["H"]
    K = data["K"]
    B = data["B"]

    all_pass = True
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

                    for i_k in range(NK):
                        ks = i_k * 32
                        ke = ks + 32

                        # dq_out should == q (bf16 exact)
                        expected_q = q[0, bos + i_ti : bos + end_r, i_h, ks:ke]
                        actual_q = data["dq_out"][0, bos + i_ti : bos + end_r, i_h, ks:ke]
                        if not torch.equal(expected_q, actual_q):
                            print(f"  FAIL dq_out: n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}")
                            print(f"    max diff: {(expected_q.float() - actual_q.float()).abs().max().item()}")
                            all_pass = False

                        # dk_out should == k (bf16 exact)
                        expected_k = k[0, bos + i_ti : bos + end_r, i_h, ks:ke]
                        actual_k = data["dk_out"][0, bos + i_ti : bos + end_r, i_h, ks:ke]
                        if not torch.equal(expected_k, actual_k):
                            print(f"  FAIL dk_out: n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}")
                            print(f"    max diff: {(expected_k.float() - actual_k.float()).abs().max().item()}")
                            all_pass = False

                        # dg_out should == g (fp32 exact)
                        expected_g = g[0, bos + i_ti : bos + end_r, i_h, ks:ke]
                        actual_g = data["dg_out"][0, bos + i_ti : bos + end_r, i_h, ks:ke]
                        if not torch.equal(expected_g, actual_g):
                            print(f"  FAIL dg_out: n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}")
                            print(f"    max diff: {(expected_g - actual_g).abs().max().item()}")
                            all_pass = False

                        # db_out[i_k] should == beta.float() (bf16→float exact)
                        expected_b = beta[0, bos + i_ti : bos + end_r, i_h].float()
                        actual_b = data["db_out"][i_k, 0, bos + i_ti : bos + end_r, i_h]
                        if not torch.equal(expected_b, actual_b):
                            print(f"  FAIL db_out: n={i_n} h={i_h} t={i_t} i={i_i} k={i_k}")
                            print(f"    max diff: {(expected_b - actual_b).abs().max().item()}")
                            all_pass = False

    if all_pass:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
    return all_pass


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
