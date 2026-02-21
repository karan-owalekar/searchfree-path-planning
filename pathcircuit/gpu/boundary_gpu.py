"""GPU-accelerated boundary distance table construction.

One CUDA block per source boundary cell, with cooperative threads.
d_arr lives in shared memory for fast access; node expansion uses
a two-pass gather-then-scatter for correctness under parallelism.
"""
from __future__ import annotations

import numpy as np
import cupy as cp

SENTINEL = 536_870_912

_BOUNDARY_TABLE_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void build_boundary_table(
    const int*       __restrict__ all_D,
    const long long* __restrict__ boundary_flat,
    const int*       __restrict__ tree_D,
    const long long* __restrict__ tree_Do,
    const long long* __restrict__ tree_cf,
    const long long* __restrict__ tree_co,
    const int*       __restrict__ tree_sz,
    const int*       __restrict__ chain_flat,
    const int*       __restrict__ chain_off,
    const int*       __restrict__ remain_flat,
    const int*       __restrict__ remain_off,
    const int*       __restrict__ BI,
    int              n_boundary,
    int              NN,
    int*             __restrict__ out
) {
    const int SENT = 536870912;
    int src_bid = blockIdx.x;
    if (src_bid >= n_boundary) return;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    int block_idx = src_bid / 12;
    int b         = src_bid % 12;
    int base_bid  = block_idx * 12;

    extern __shared__ int smem[];
    int* d        = smem;             /* [NN]  */
    int* bi_sh    = smem + NN;        /* [512] */
    int* bd_sh    = bi_sh + 512;      /* [512] */
    /* total shared: (NN + 1024) * 4 bytes */

    /* ---- initialise d ---- */
    for (int i = tid; i < NN; i += nthreads) d[i] = SENT;
    __syncthreads();

    /* ---- seed from base-block D ---- */
    if (tid == 0) {
        int src_local = BI[b];
        const int* D_blk = &all_D[block_idx * 256];
        for (int j = 0; j < 12; j++) {
            int v = D_blk[src_local * 16 + BI[j]];
            if (v < SENT) d[boundary_flat[base_bid + j]] = v;
        }
    }
    __syncthreads();

    /* ---- expand one batch of nodes ---- */
#define EXPAND_NODES(idx_arr, start, end) \
    for (int ii = (start); ii < (end); ii++) { \
        int ni  = (idx_arr)[ii]; \
        int nsz = tree_sz[ni]; \
        if (nsz == 0) continue; \
        long long co  = tree_co[ni]; \
        long long dof = tree_Do[ni]; \
        \
        /* Phase 1: gather known cells (parallel, atomic counter) */ \
        __shared__ int n_br; \
        if (tid == 0) n_br = 0; \
        __syncthreads(); \
        for (int i = tid; i < nsz; i += nthreads) { \
            int ci = (int)tree_cf[co + i]; \
            int cv = d[ci]; \
            if (cv < SENT) { \
                int pos = atomicAdd(&n_br, 1); \
                bi_sh[pos] = i; \
                bd_sh[pos] = cv; \
            } \
        } \
        __syncthreads(); \
        \
        if (n_br == 0) continue; \
        \
        /* Phase 2: update target cells (parallel over k) */ \
        for (int k = tid; k < nsz; k += nthreads) { \
            int ck   = (int)tree_cf[co + k]; \
            int best = d[ck]; \
            int nbr  = n_br; \
            for (int bb = 0; bb < nbr; bb++) { \
                int dg = tree_D[dof + (long long)bi_sh[bb] * nsz + k]; \
                if (dg >= SENT) continue; \
                int val = bd_sh[bb] + dg; \
                if (val < best) best = val; \
            } \
            d[ck] = best; \
        } \
        __syncthreads(); \
    }

    EXPAND_NODES(chain_flat,  chain_off[block_idx],  chain_off[block_idx + 1])
    EXPAND_NODES(remain_flat, remain_off[block_idx], remain_off[block_idx + 1])

#undef EXPAND_NODES

    /* ---- extract boundary distances ---- */
    for (int bid2 = tid; bid2 < n_boundary; bid2 += nthreads) {
        int v = d[(int)boundary_flat[bid2]];
        if (v < SENT)
            out[(long long)src_bid * n_boundary + bid2] = v;
    }
}
""", "build_boundary_table")


def gpu_build_boundary_table(
    all_D: np.ndarray,
    boundary_flat: np.ndarray,
    tree_data: tuple,
    chain_indices: dict,
    remaining_indices: dict,
    blocks_r: int,
    blocks_c: int,
    N: int,
) -> np.ndarray:
    """Build boundary distance table on GPU.  Returns int32[n_boundary, n_boundary]."""
    from pathcircuit.core.base_block import BOUNDARY_INDICES

    tree_D_flat, tree_Do, tree_cf, tree_co, tree_sz, tree_n = tree_data

    n_blocks = blocks_r * blocks_c
    n_boundary = n_blocks * 12
    NN = N * N

    chain_parts, chain_offs = [], [0]
    remain_parts, remain_offs = [], [0]
    empty = np.empty(0, dtype=np.int32)

    for br in range(blocks_r):
        for bc in range(blocks_c):
            ci = chain_indices.get((br, bc), empty)
            ri = remaining_indices.get((br, bc), empty)
            chain_parts.append(ci)
            chain_offs.append(chain_offs[-1] + len(ci))
            remain_parts.append(ri)
            remain_offs.append(remain_offs[-1] + len(ri))

    chain_flat_np = np.concatenate(chain_parts).astype(np.int32) if chain_offs[-1] > 0 else np.zeros(1, dtype=np.int32)
    remain_flat_np = np.concatenate(remain_parts).astype(np.int32) if remain_offs[-1] > 0 else np.zeros(1, dtype=np.int32)

    d_all_D = cp.asarray(all_D.ravel().astype(np.int32))
    d_bf = cp.asarray(boundary_flat.astype(np.int64))
    d_tree_D = cp.asarray(tree_D_flat.astype(np.int32))
    d_tree_Do = cp.asarray(tree_Do.astype(np.int64))
    d_tree_cf = cp.asarray(tree_cf.astype(np.int64))
    d_tree_co = cp.asarray(tree_co.astype(np.int64))
    d_tree_sz = cp.asarray(tree_sz.astype(np.int32))
    d_chain = cp.asarray(chain_flat_np)
    d_chain_off = cp.asarray(np.array(chain_offs, dtype=np.int32))
    d_remain = cp.asarray(remain_flat_np)
    d_remain_off = cp.asarray(np.array(remain_offs, dtype=np.int32))
    d_BI = cp.asarray(np.array(BOUNDARY_INDICES, dtype=np.int32))

    d_out = cp.full(n_boundary * n_boundary, SENTINEL, dtype=cp.int32)

    threads_per_block = 32
    smem_bytes = (NN + 1024) * 4

    _BOUNDARY_TABLE_KERNEL(
        (n_boundary,), (threads_per_block,),
        (d_all_D, d_bf, d_tree_D, d_tree_Do, d_tree_cf, d_tree_co, d_tree_sz,
         d_chain, d_chain_off, d_remain, d_remain_off, d_BI,
         np.int32(n_boundary), np.int32(NN), d_out),
        shared_mem=smem_bytes,
    )

    cp.cuda.stream.get_current_stream().synchronize()
    result = d_out.get().reshape(n_boundary, n_boundary)

    del d_out, d_all_D, d_bf, d_tree_D, d_tree_Do, d_tree_cf
    del d_tree_co, d_tree_sz, d_chain, d_chain_off, d_remain, d_remain_off

    return result
