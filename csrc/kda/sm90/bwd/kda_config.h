#pragma once

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
    void* __restrict__ db2_ptr;            //[NK, total_q_len, h] - per-K-tile db partials
    void* __restrict__ dg_out_ptr;         //[b, t, h, d]
    void* __restrict__ cu_seqlens_ptr;     //[b + 1]
    void* __restrict__ chunk_indices_ptr;  //[num_chunks, 2]

    int num_chunks;
    int num_k_tiles;
    void* tile_counter_ptr;
};
