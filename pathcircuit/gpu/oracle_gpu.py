"""GPU-accelerated oracle path reconstruction using the successor-field architecture.

Layer 0: Gateway attention (distance query) — runs on CPU (144 lookups, ~5μs)
Layer 1: d(cell, g) for ALL cells — one CUDA kernel, one thread per cell
Layer 2: Successor field — one CUDA kernel, one thread per cell
Layer 3: Path assembly — CPU pointer chase (zero-cost per step)
"""
from __future__ import annotations

import time
import numpy as np
import cupy as cp

SENTINEL = 536_870_912

# -----------------------------------------------------------------------
# CUDA kernel: compute d(cell, goal) for every cell on the grid
# -----------------------------------------------------------------------
_DIST_TO_GOAL_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void dist_to_goal(
    const int*   __restrict__ all_D,         // [n_blocks * 16 * 16] flattened
    const int*   __restrict__ boundary_dist, // [n_bnd * n_bnd] flattened
    const int*   __restrict__ block_bids,    // [n_blocks * 12]
    const int*   __restrict__ BI,            // [12]
    const long long* __restrict__ vg_d,      // [n_vg]
    const int*   __restrict__ vg_bid,        // [n_vg]
    int n_vg, int n_bnd,
    int g_br, int g_bc, int g_local,
    int blocks_c,
    int rows, int cols,
    long long*   __restrict__ dist_out       // [rows * cols]
) {
    const long long SENT = 536870912LL;
    int block_idx = blockIdx.x;
    int cell      = threadIdx.x;
    if (cell >= 16) return;

    int br = block_idx / blocks_c;
    int bc = block_idx % blocks_c;

    // Shared memory: mid-distances for this block (12 × n_vg, max 12×12)
    __shared__ long long mid[12][12];

    // First 12 threads cooperatively load one row each
    if (cell < 12) {
        int ba = block_bids[block_idx * 12 + cell];
        for (int vb = 0; vb < n_vg && vb < 12; vb++) {
            mid[cell][vb] = (long long)boundary_dist[(long long)ba * n_bnd + vg_bid[vb]];
        }
    }
    __syncthreads();

    const int* D_cell = &all_D[block_idx * 256 + cell * 16];
    long long best = SENT;

    for (int a = 0; a < 12; a++) {
        long long d_ca = (long long)D_cell[BI[a]];
        if (d_ca >= SENT) continue;
        for (int vb = 0; vb < n_vg && vb < 12; vb++) {
            long long m = mid[a][vb];
            if (m >= SENT) continue;
            long long val = d_ca + m + vg_d[vb];
            if (val < best) best = val;
        }
    }

    if (br == g_br && bc == g_bc) {
        long long ld = (long long)D_cell[g_local];
        if (ld < best) best = ld;
    }

    int cr = br * 4 + cell / 4;
    int cc = bc * 4 + cell % 4;
    dist_out[cr * cols + cc] = best;
}
""", "dist_to_goal")

# -----------------------------------------------------------------------
# CUDA kernel: build successor field
# -----------------------------------------------------------------------
_SUCCESSOR_KERNEL_8 = cp.RawKernel(r"""
extern "C" __global__
void build_successor_8(
    const long long* __restrict__ costs,
    const long long* __restrict__ dist_to_goal,
    int*             __restrict__ successor,
    int rows, int cols
) {
    const int SENT = 536870912;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    long long cu = costs[idx];
    if (cu >= SENT) { successor[idx] = -1; return; }

    int r = idx / cols;
    int c = idx % cols;

    const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[8] = {-1,  0,  1,-1, 1,-1, 0, 1};

    int   best_next  = -1;
    long long best_score = SENT;

    for (int k = 0; k < 8; k++) {
        int nr = r + dr[k];
        int nc = c + dc[k];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        int nidx = nr * cols + nc;
        long long cn = costs[nidx];
        if (cn >= SENT) continue;
        long long score = cu + cn + dist_to_goal[nidx];
        if (score < best_score) {
            best_score = score;
            best_next  = nidx;
        }
    }
    successor[idx] = best_next;
}
""", "build_successor_8")

_SUCCESSOR_KERNEL_4 = cp.RawKernel(r"""
extern "C" __global__
void build_successor_4(
    const long long* __restrict__ costs,
    const long long* __restrict__ dist_to_goal,
    int*             __restrict__ successor,
    int rows, int cols
) {
    const int SENT = 536870912;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    long long cu = costs[idx];
    if (cu >= SENT) { successor[idx] = -1; return; }

    int r = idx / cols;
    int c = idx % cols;

    const int dr[4] = {-1, 0, 0, 1};
    const int dc[4] = { 0,-1, 1, 0};

    int   best_next  = -1;
    long long best_score = SENT;

    for (int k = 0; k < 4; k++) {
        int nr = r + dr[k];
        int nc = c + dc[k];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        int nidx = nr * cols + nc;
        long long cn = costs[nidx];
        if (cn >= SENT) continue;
        long long score = cu + cn + dist_to_goal[nidx];
        if (score < best_score) {
            best_score = score;
            best_next  = nidx;
        }
    }
    successor[idx] = best_next;
}
""", "build_successor_4")


# -----------------------------------------------------------------------
# GPU data cache — avoids re-transferring oracle data on every query
# -----------------------------------------------------------------------

_gpu_cache: dict = {}


def _get_gpu_data(oracle):
    """Upload oracle arrays to GPU once; reuse across queries."""
    cache_key = id(oracle)
    if cache_key in _gpu_cache:
        return _gpu_cache[cache_key]

    from pathcircuit.core.base_block import BOUNDARY_INDICES

    blocks_r = len(oracle.base_blocks)
    blocks_c = len(oracle.base_blocks[0])
    n_blocks = blocks_r * blocks_c
    n_bnd = oracle.boundary_dist.shape[0]

    data = {
        "d_all_D": cp.asarray(oracle.all_D.ravel().astype(np.int32)),
        "d_bd": cp.asarray(oracle.boundary_dist),
        "d_bids": cp.asarray(oracle.block_bids.reshape(n_blocks, 12).astype(np.int32)),
        "d_BI": cp.asarray(np.array(BOUNDARY_INDICES, dtype=np.int32)),
        "d_costs": cp.asarray(oracle.costs_flat),
        "n_bnd": n_bnd,
        "n_blocks": n_blocks,
        "blocks_r": blocks_r,
        "blocks_c": blocks_c,
    }
    _gpu_cache.clear()
    _gpu_cache[cache_key] = data
    return data


def invalidate_gpu_cache():
    """Call when the oracle changes (new grid / obstacles)."""
    _gpu_cache.clear()


# -----------------------------------------------------------------------
# Host-side orchestration
# -----------------------------------------------------------------------

def gpu_reconstruct_path(
    oracle,
    s: tuple[int, int],
    g: tuple[int, int],
) -> tuple[float, list[tuple[int, int]]]:
    """GPU-accelerated path reconstruction via the successor-field architecture.

    Returns (elapsed_ms, path).
    """
    from pathcircuit.core.base_block import BOUNDARY_INDICES

    sr, sc = s
    gr, gc = g
    grid = oracle.grid
    rows, cols = grid.rows, grid.cols

    gd = _get_gpu_data(oracle)
    blocks_c = gd["blocks_c"]
    n_blocks = gd["n_blocks"]
    n_bnd = gd["n_bnd"]

    g_br, g_bc = gr // 4, gc // 4
    g_local = (gr % 4) * 4 + (gc % 4)

    D_g = oracle.all_D[g_br * blocks_c + g_bc]

    n_vg = 0
    vg_d_host = np.empty(12, dtype=np.int64)
    vg_bid_host = np.empty(12, dtype=np.int32)
    for b in range(12):
        d = int(D_g[BOUNDARY_INDICES[b], g_local])
        if d < SENTINEL:
            vg_d_host[n_vg] = d
            vg_bid_host[n_vg] = oracle.block_bids[g_br, g_bc, b]
            n_vg += 1

    t0 = time.perf_counter()

    # Only goal-specific data transferred per query
    d_vg_d = cp.asarray(vg_d_host[:n_vg])
    d_vg_bid = cp.asarray(vg_bid_host[:n_vg].astype(np.int32))
    d_dist = cp.full(rows * cols, SENTINEL, dtype=cp.int64)
    d_succ = cp.full(rows * cols, -1, dtype=cp.int32)

    # --- Layer 1: d(cell, g) for all cells ---
    _DIST_TO_GOAL_KERNEL(
        (n_blocks,), (16,),
        (gd["d_all_D"], gd["d_bd"], gd["d_bids"], gd["d_BI"],
         d_vg_d, d_vg_bid, np.int32(n_vg), np.int32(n_bnd),
         np.int32(g_br), np.int32(g_bc), np.int32(g_local),
         np.int32(blocks_c),
         np.int32(rows), np.int32(cols),
         d_dist),
    )

    # --- Layer 2: successor field ---
    threads = 256
    grid_size = (rows * cols + threads - 1) // threads
    kern = _SUCCESSOR_KERNEL_8 if grid.connectivity == 8 else _SUCCESSOR_KERNEL_4
    kern(
        (grid_size,), (threads,),
        (gd["d_costs"], d_dist, d_succ, np.int32(rows), np.int32(cols)),
    )

    # --- Copy back ---
    successor = d_succ.get()
    cp.cuda.stream.get_current_stream().synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000.0

    # --- Layer 3: follow pointers (CPU) ---
    path = [(sr, sc)]
    cur = sr * cols + sc
    goal_idx = gr * cols + gc
    max_steps = rows * cols
    while cur != goal_idx and len(path) < max_steps:
        nxt = int(successor[cur])
        if nxt < 0:
            break
        path.append((nxt // cols, nxt % cols))
        cur = nxt

    return gpu_ms, path
