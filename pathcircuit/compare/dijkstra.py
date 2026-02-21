from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np

from pathcircuit.grid import Grid, SENTINEL

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


@dataclass
class DijkstraResult:
    path: list[tuple[int, int]]
    cost: int
    time_ms: float
    nodes_expanded: int


# ---------------------------------------------------------------------------
# Numba JIT kernel — array-based binary min-heap, no Python objects
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @_njit(cache=True)
    def _dijkstra_core(costs, rows, cols, connectivity,  # type: ignore[misc]
                       start_idx, goal_idx):
        _S = np.int64(536_870_912)
        n = rows * cols

        dist = np.full(n, _S, dtype=np.int64)
        prev_arr = np.full(n, -1, dtype=np.int32)

        if costs[start_idx] >= _S or costs[goal_idx] >= _S:
            return dist, prev_arr, 0

        dist[start_idx] = np.int64(0)

        cap = n * 8
        h_pri = np.empty(cap, dtype=np.int64)
        h_node = np.empty(cap, dtype=np.int32)
        h_size = 0

        h_pri[0] = np.int64(0)
        h_node[0] = np.int32(start_idx)
        h_size = 1

        expanded = 0

        if connectivity == 8:
            dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:
            dr = np.array([-1, 0, 0, 1], dtype=np.int32)
            dc = np.array([0, -1, 1, 0], dtype=np.int32)

        n_nbrs = len(dr)

        while h_size > 0:
            d_u = h_pri[0]
            u = h_node[0]
            h_size -= 1
            if h_size > 0:
                h_pri[0] = h_pri[h_size]
                h_node[0] = h_node[h_size]
                pos = 0
                while True:
                    left = 2 * pos + 1
                    right = 2 * pos + 2
                    sm = pos
                    if left < h_size and h_pri[left] < h_pri[sm]:
                        sm = left
                    if right < h_size and h_pri[right] < h_pri[sm]:
                        sm = right
                    if sm == pos:
                        break
                    h_pri[pos], h_pri[sm] = h_pri[sm], h_pri[pos]
                    h_node[pos], h_node[sm] = h_node[sm], h_node[pos]
                    pos = sm

            if d_u > dist[u]:
                continue

            expanded += 1
            if u == goal_idx:
                break

            ur = u // cols
            uc = u % cols
            cu = np.int64(costs[u])

            for k in range(n_nbrs):
                nr = ur + dr[k]
                nc = uc + dc[k]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                v = nr * cols + nc
                cv = np.int64(costs[v])
                if cv >= _S:
                    continue
                nd = d_u + cu + cv
                if nd < dist[v]:
                    dist[v] = nd
                    prev_arr[v] = np.int32(u)
                    if h_size < cap:
                        p = h_size
                        h_pri[p] = nd
                        h_node[p] = np.int32(v)
                        h_size += 1
                        while p > 0:
                            par = (p - 1) // 2
                            if h_pri[p] < h_pri[par]:
                                h_pri[p], h_pri[par] = h_pri[par], h_pri[p]
                                h_node[p], h_node[par] = h_node[par], h_node[p]
                                p = par
                            else:
                                break

        return dist, prev_arr, expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dijkstra(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> DijkstraResult:
    sr, sc = start
    gr, gc = goal

    if grid.is_obstacle(sr, sc) or grid.is_obstacle(gr, gc):
        return DijkstraResult(path=[], cost=SENTINEL, time_ms=0.0, nodes_expanded=0)

    t0 = time.perf_counter()

    if _HAS_NUMBA:
        costs = grid.costs.ravel().astype(np.int64)
        start_idx = sr * grid.cols + sc
        goal_idx = gr * grid.cols + gc

        dist, prev_arr, expanded = _dijkstra_core(
            costs, grid.rows, grid.cols, grid.connectivity,
            start_idx, goal_idx,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        if dist[goal_idx] >= SENTINEL:
            return DijkstraResult(path=[], cost=SENTINEL, time_ms=elapsed,
                                  nodes_expanded=int(expanded))

        path: list[tuple[int, int]] = []
        cur = goal_idx
        while cur != -1:
            path.append((cur // grid.cols, cur % grid.cols))
            cur = int(prev_arr[cur])
        path.reverse()

        return DijkstraResult(path=path, cost=int(dist[goal_idx]),
                              time_ms=elapsed, nodes_expanded=int(expanded))

    # ---- pure-Python fallback (no numba) ----
    dist_d: dict[tuple[int, int], int] = {start: 0}
    prev_d: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[int, tuple[int, int]]] = [(0, start)]
    expanded_py = 0

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist_d.get(u, SENTINEL):
            continue
        expanded_py += 1
        if u == goal:
            break
        for nr, nc in grid.get_neighbors(u[0], u[1]):
            w = grid.edge_weight(u[0], u[1], nr, nc)
            if w >= SENTINEL:
                continue
            nd = d + w
            v = (nr, nc)
            if nd < dist_d.get(v, SENTINEL):
                dist_d[v] = nd
                prev_d[v] = u
                heapq.heappush(heap, (nd, v))

    elapsed = (time.perf_counter() - t0) * 1000.0

    if goal not in dist_d:
        return DijkstraResult(path=[], cost=SENTINEL, time_ms=elapsed,
                              nodes_expanded=expanded_py)

    path_py: list[tuple[int, int]] = []
    cur_py: tuple[int, int] | None = goal
    while cur_py is not None:
        path_py.append(cur_py)
        cur_py = prev_d.get(cur_py)
    path_py.reverse()

    return DijkstraResult(path=path_py, cost=dist_d[goal], time_ms=elapsed,
                          nodes_expanded=expanded_py)
