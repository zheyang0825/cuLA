#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

struct NaiveTileScheduler {
    struct Params {
        int num_chunks;  // total chunk tiles: sum(ceil(seq_len_i / BT))
        int num_heads;   // H
        int num_k;       // NK = K / BK
        int num_c;       // NC = BT / BC
    };

    Params params;

    CUTLASS_DEVICE
    NaiveTileScheduler(Params const& params) : params(params) {
    }

    // Grid: x = num_k * num_c,  y = num_chunks,  z = num_heads
    static dim3
    get_grid_shape(Params const& params) {
        return dim3(params.num_k * params.num_c, params.num_chunks, params.num_heads);
    }

    // blockIdx.x = i_k * num_c + i_i,  blockIdx.y = tile_id,  blockIdx.z = i_h
    CUTLASS_DEVICE
    auto
    get_block_coord(const int* chunk_indices_ptr, const int* cu_seqlens_ptr) {
        using namespace cute;
        const int i_kc = blockIdx.x;
        const int tile_id = blockIdx.y;
        const int i_h = blockIdx.z;
        const int i_k = i_kc / params.num_c;
        const int i_i = i_kc % params.num_c;
        const int i_n = chunk_indices_ptr[tile_id * 2];
        const int i_t = chunk_indices_ptr[tile_id * 2 + 1];
        const int bos = cu_seqlens_ptr[i_n];
        const int eos = cu_seqlens_ptr[i_n + 1];

        return make_coord(i_n, i_h, i_t, i_k, i_i, bos, eos - bos);
    }
};