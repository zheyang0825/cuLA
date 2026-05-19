from typing import Tuple, Optional
import torch

from kda.kda_cuda import chunk_kda_bwd_intra_cuda

def kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    db_out: torch.Tensor,
    dg_out: torch.Tensor,
    chunk_size: int,
    tile_counter: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    chunk_kda_bwd_intra_cuda(q, k, g, beta.float(), dAqk, dAkk, dq, dk, db, dg,
                             cu_seqlens.to(torch.int32), chunk_indices,
                             dq_out, dk_out, db_out, dg_out, chunk_size)
    return dq_out, dk_out, db_out, dg_out
