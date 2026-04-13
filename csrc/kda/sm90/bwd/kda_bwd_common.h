#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

// =====================================================================
// StaticPersistentTileScheduler
// No smem synchronization needed — every CTA processes tiles starting
// at blockIdx.x and striding by gridDim.x. All warps within a CTA
// independently maintain the same tile_id, so no tile pipeline is needed.
// =====================================================================
struct StaticPersistentTileScheduler {
    struct Params {
        int num_blocks;  // number of sequence chunks (from chunk_indices)
        int num_heads;

        int num_sm;
        int* tile_counter;  // unused
    };

    int current_tile_id;
    Params params;

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(Params const& params) : current_tile_id(blockIdx.x), params(params) {
    }

    static dim3
    get_grid_shape(Params const& params) {
        dim3 grid(params.num_sm, 1, 1);
        return grid;
    }

    CUTLASS_DEVICE
    int
    total_tiles() const {
        return params.num_blocks * params.num_heads;
    }

    CUTLASS_DEVICE
    int
    get_current_tile_id() const {
        return current_tile_id;
    }

    CUTLASS_DEVICE
    void
    advance() {
        current_tile_id += gridDim.x;
    }

    CUTLASS_DEVICE
    bool
    is_valid() const {
        return current_tile_id < total_tiles();
    }

    CUTLASS_DEVICE
    static auto
    decode_tile_coord(int tile_id, int num_heads, int* chunk_indices_ptr, int* cu_seqlens_ptr) {
        using namespace cute;
        int tile_idx_raw = tile_id / num_heads;
        int head_idx = tile_id % num_heads;
        int batch_idx = chunk_indices_ptr[tile_idx_raw * 2];
        int seq_idx = chunk_indices_ptr[tile_idx_raw * 2 + 1];
        return make_coord(batch_idx, head_idx, seq_idx, 0);
    }
};

// =====================================================================
// KDA_bwd_intra_params — aligned with SM100 kda_config.h
// =====================================================================
struct KDA_bwd_intra_params {
    using index_t = int64_t;

    int total_q_len;
    int b;
    int h;
    int d, d_v;
    int chunk_size;

    void* __restrict__ q_ptr;              //[b, t, h, d]
    void* __restrict__ k_ptr;              //[b, t, h, d]
    void* __restrict__ g_ptr;              //[b, t, h, d]
    void* __restrict__ beta_ptr;           //[b, t, h]
    void* __restrict__ dAqk_ptr;           //[b, t, h, BT]
    void* __restrict__ dAkk_ptr;           //[b, t, h, BT]
    void* __restrict__ dq_ptr;             //[b, t, h, d]
    void* __restrict__ dk_ptr;             //[b, t, h, d]
    void* __restrict__ db_ptr;             //[b, t, h]
    void* __restrict__ dg_ptr;             //[b, t, h, d]
    void* __restrict__ dq_out_ptr;         //[b, t, h, d]
    void* __restrict__ dk_out_ptr;         //[b, t, h, d]
    void* __restrict__ db_out_ptr;         //[b, t, h]
    void* __restrict__ dg_out_ptr;         //[b, t, h, d]
    void* __restrict__ cu_seqlens_ptr;     //[b + 1]
    void* __restrict__ chunk_indices_ptr;  //[len(cu_seqlens) - 1]

    StaticPersistentTileScheduler::Params tile_scheduler_params;

    int num_sm;
};
