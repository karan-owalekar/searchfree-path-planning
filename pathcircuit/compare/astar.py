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
class AStarResult:
    path: list[tuple[int, int]]
    cost: int
    time_ms: float
    nodes_expanded: int


# ---------------------------------------------------------------------------
# Numba JIT kernel — array-based binary min-heap with inline heuristic
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @_njit(cache=True)
    def _astar_core(costs, rows, cols, connectivity,  # type: ignore[misc]
                    start_idx, goal_idx):
        _S = np.int64(536_870_912)
        n = rows * cols

        g_score = np.full(n, _S, dtype=np.int64)
        prev_arr = np.full(n, -1, dtype=np.int32)

        if costs[start_idx] >= _S or costs[goal_idx] >= _S:
            return g_score, prev_arr, 0

        goal_r = goal_idx // cols
        goal_c = goal_idx % cols

        g_score[start_idx] = np.int64(0)
        sr = start_idx // cols
        sc = start_idx % cols
        if connectivity == 8:
            h0 = np.int64(2 * max(abs(sr - goal_r), abs(sc - goal_c)))
        else:
            h0 = np.int64(2 * (abs(sr - goal_r) + abs(sc - goal_c)))

        cap = n * 8
        h_f = np.empty(cap, dtype=np.int64)
        h_node = np.empty(cap, dtype=np.int32)
        h_size = 0

        h_f[0] = h0
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
            f_u = h_f[0]
            u = h_node[0]
            h_size -= 1
            if h_size > 0:
                h_f[0] = h_f[h_size]
                h_node[0] = h_node[h_size]
                pos = 0
                while True:
                    left = 2 * pos + 1
                    right = 2 * pos + 2
                    sm = pos
                    if left < h_size and h_f[left] < h_f[sm]:
                        sm = left
                    if right < h_size and h_f[right] < h_f[sm]:
                        sm = right
                    if sm == pos:
                        break
                    h_f[pos], h_f[sm] = h_f[sm], h_f[pos]
                    h_node[pos], h_node[sm] = h_node[sm], h_node[pos]
                    pos = sm

            g_u = g_score[u]
            ur = u // cols
            uc = u % cols
            if connectivity == 8:
                h_u = np.int64(2 * max(abs(ur - goal_r), abs(uc - goal_c)))
            else:
                h_u = np.int64(2 * (abs(ur - goal_r) + abs(uc - goal_c)))
            if f_u > g_u + h_u:
                continue

            expanded += 1
            if u == goal_idx:
                break

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
                ng = g_u + cu + cv
                if ng < g_score[v]:
                    g_score[v] = ng
                    prev_arr[v] = np.int32(u)
                    if connectivity == 8:
                        h_v = np.int64(2 * max(abs(nr - goal_r), abs(nc - goal_c)))
                    else:
                        h_v = np.int64(2 * (abs(nr - goal_r) + abs(nc - goal_c)))
                    fv = ng + h_v
                    if h_size < cap:
                        p = h_size
                        h_f[p] = fv
                        h_node[p] = np.int32(v)
                        h_size += 1
                        while p > 0:
                            par = (p - 1) // 2
                            if h_f[p] < h_f[par]:
                                h_f[p], h_f[par] = h_f[par], h_f[p]
                                h_node[p], h_node[par] = h_node[par], h_node[p]
                                p = par
                            else:
                                break

        return g_score, prev_arr, expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> AStarResult:
    sr, sc = start
    gr, gc = goal

    if grid.is_obstacle(sr, sc) or grid.is_obstacle(gr, gc):
        return AStarResult(path=[], cost=SENTINEL, time_ms=0.0, nodes_expanded=0)

    t0 = time.perf_counter()

    if _HAS_NUMBA:
        costs = grid.costs.ravel().astype(np.int64)
        start_idx = sr * grid.cols + sc
        goal_idx = gr * grid.cols + gc

        g_score, prev_arr, expanded = _astar_core(
            costs, grid.rows, grid.cols, grid.connectivity,
            start_idx, goal_idx,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        if g_score[goal_idx] >= SENTINEL:
            return AStarResult(path=[], cost=SENTINEL, time_ms=elapsed,
                               nodes_expanded=int(expanded))

        path: list[tuple[int, int]] = []
        cur = goal_idx
        while cur != -1:
            path.append((cur // grid.cols, cur % grid.cols))
            cur = int(prev_arr[cur])
        path.reverse()

        return AStarResult(path=path, cost=int(g_score[goal_idx]),
                           time_ms=elapsed, nodes_expanded=int(expanded))

    # ---- pure-Python fallback (no numba) ----
    g_score_d: dict[tuple[int, int], int] = {start: 0}
    prev_d: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    conn = grid.connectivity
    dr_abs = abs(sr - gr)
    dc_abs = abs(sc - gc)
    h0 = 2 * max(dr_abs, dc_abs) if conn == 8 else 2 * (dr_abs + dc_abs)
    counter = 0
    heap: list[tuple[int, int, tuple[int, int]]] = [(h0, counter, start)]
    expanded_py = 0

    while heap:
        f, _, u = heapq.heappop(heap)
        g_u = g_score_d.get(u, SENTINEL)
        dr_abs = abs(u[0] - gr)
        dc_abs = abs(u[1] - gc)
        h_u = 2 * max(dr_abs, dc_abs) if conn == 8 else 2 * (dr_abs + dc_abs)
        if f > g_u + h_u:
            continue
        expanded_py += 1
        if u == goal:
            break
        for nr, nc in grid.get_neighbors(u[0], u[1]):
            w = grid.edge_weight(u[0], u[1], nr, nc)
            if w >= SENTINEL:
                continue
            ng = g_u + w
            v = (nr, nc)
            if ng < g_score_d.get(v, SENTINEL):
                g_score_d[v] = ng
                prev_d[v] = u
                counter += 1
                dr_abs = abs(nr - gr)
                dc_abs = abs(nc - gc)
                h_v = 2 * max(dr_abs, dc_abs) if conn == 8 else 2 * (dr_abs + dc_abs)
                heapq.heappush(heap, (ng + h_v, counter, v))

    elapsed = (time.perf_counter() - t0) * 1000.0

    if goal not in g_score_d:
        return AStarResult(path=[], cost=SENTINEL, time_ms=elapsed,
                           nodes_expanded=expanded_py)

    path_py: list[tuple[int, int]] = []
    cur_py: tuple[int, int] | None = goal
    while cur_py is not None:
        path_py.append(cur_py)
        cur_py = prev_d.get(cur_py)
    path_py.reverse()

    return AStarResult(path=path_py, cost=g_score_d[goal], time_ms=elapsed,
                       nodes_expanded=expanded_py)
