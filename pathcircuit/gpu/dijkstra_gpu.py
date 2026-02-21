"""GPU-accelerated shortest path via parallel Bellman-Ford relaxation.

Each iteration relaxes ALL edges from ALL active cells in parallel.
Converges in O(diameter) iterations — each iteration is one CUDA kernel launch
processing every cell simultaneously.

For grids this is equivalent to a wavefront BFS weighted by edge costs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import cupy as cp

SENTINEL = 536_870_912


@dataclass
class GPUDijkstraResult:
    cost: int
    path: list[tuple[int, int]]
    time_ms: float
    iterations: int


_RELAX_KERNEL_8 = cp.RawKernel(r"""
extern "C" __global__
void relax_8(
    const long long* __restrict__ costs,
    long long*       __restrict__ dist,
    int*             __restrict__ changed,
    int rows, int cols
) {
    const long long SENT = 536870912LL;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    long long d_cell = dist[idx];
    if (d_cell >= SENT) return;

    long long cu = costs[idx];
    if (cu >= SENT) return;

    int r = idx / cols;
    int c = idx % cols;

    const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[8] = {-1,  0,  1,-1, 1,-1, 0, 1};

    for (int k = 0; k < 8; k++) {
        int nr = r + dr[k];
        int nc = c + dc[k];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        int nidx = nr * cols + nc;
        long long cn = costs[nidx];
        if (cn >= SENT) continue;
        long long new_dist = d_cell + cu + cn;
        long long old = atomicMin((unsigned long long*)&dist[nidx],
                                  (unsigned long long)new_dist);
        if (new_dist < old) {
            changed[0] = 1;
        }
    }
}
""", "relax_8")

_RELAX_KERNEL_4 = cp.RawKernel(r"""
extern "C" __global__
void relax_4(
    const long long* __restrict__ costs,
    long long*       __restrict__ dist,
    int*             __restrict__ changed,
    int rows, int cols
) {
    const long long SENT = 536870912LL;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    long long d_cell = dist[idx];
    if (d_cell >= SENT) return;

    long long cu = costs[idx];
    if (cu >= SENT) return;

    int r = idx / cols;
    int c = idx % cols;

    const int dr[4] = {-1, 0, 0, 1};
    const int dc[4] = { 0,-1, 1, 0};

    for (int k = 0; k < 4; k++) {
        int nr = r + dr[k];
        int nc = c + dc[k];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        int nidx = nr * cols + nc;
        long long cn = costs[nidx];
        if (cn >= SENT) continue;
        long long new_dist = d_cell + cu + cn;
        long long old = atomicMin((unsigned long long*)&dist[nidx],
                                  (unsigned long long)new_dist);
        if (new_dist < old) {
            changed[0] = 1;
        }
    }
}
""", "relax_4")


def gpu_dijkstra(
    grid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> GPUDijkstraResult:
    """GPU parallel Bellman-Ford SSSP, then CPU path traceback."""
    rows, cols = grid.rows, grid.cols
    sr, sc = start
    gr, gc = goal
    total_cells = rows * cols

    t0 = time.perf_counter()

    costs_host = grid.costs.ravel().astype(np.int64)
    d_costs = cp.asarray(costs_host)

    d_dist = cp.full(total_cells, SENTINEL, dtype=cp.int64)
    d_dist[sr * cols + sc] = 0

    d_changed = cp.zeros(1, dtype=cp.int32)

    kern = _RELAX_KERNEL_8 if grid.connectivity == 8 else _RELAX_KERNEL_4
    threads = 256
    blocks = (total_cells + threads - 1) // threads

    iterations = 0
    max_iter = rows * cols
    while iterations < max_iter:
        d_changed[0] = 0
        kern((blocks,), (threads,),
             (d_costs, d_dist, d_changed, np.int32(rows), np.int32(cols)))
        iterations += 1
        if d_changed.get().item() == 0:
            break

    dist_host = d_dist.get()
    cp.cuda.stream.get_current_stream().synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    goal_cost = int(dist_host[gr * cols + gc])

    if goal_cost >= SENTINEL:
        return GPUDijkstraResult(cost=SENTINEL, path=[], time_ms=elapsed_ms,
                                 iterations=iterations)

    # Trace path on CPU
    costs_np = costs_host
    path = [(gr, gc)]
    cur_r, cur_c = gr, gc
    if grid.connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    while (cur_r, cur_c) != (sr, sc):
        d_cur = dist_host[cur_r * cols + cur_c]
        found = False
        for dr, dc in offsets:
            pr, pc = cur_r + dr, cur_c + dc
            if pr < 0 or pr >= rows or pc < 0 or pc >= cols:
                continue
            pidx = pr * cols + pc
            d_prev = dist_host[pidx]
            if d_prev >= SENTINEL:
                continue
            edge_w = int(costs_np[pidx]) + int(costs_np[cur_r * cols + cur_c])
            if d_prev + edge_w == d_cur:
                path.append((pr, pc))
                cur_r, cur_c = pr, pc
                found = True
                break
        if not found:
            break

    path.reverse()
    return GPUDijkstraResult(cost=goal_cost, path=path, time_ms=elapsed_ms,
                             iterations=iterations)
