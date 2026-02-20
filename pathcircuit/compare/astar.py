from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from pathcircuit.grid import Grid, SENTINEL


@dataclass
class AStarResult:
    path: list[tuple[int, int]]
    cost: int
    time_ms: float
    nodes_expanded: int


def _heuristic(a: tuple[int, int], b: tuple[int, int], connectivity: int) -> int:
    """Admissible heuristic scaled to match edge_weight = c(i)+c(j) with unit costs.

    Cardinal edge cost between two unit-cost cells = 2.
    Diagonal edge cost (8-conn) between two unit-cost cells = 2 (same formula).
    Manhattan/Chebyshev distance in cells * 2 gives the lower bound.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if connectivity == 8:
        return 2 * max(dr, dc)
    return 2 * (dr + dc)


def astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> AStarResult:
    sr, sc = start
    gr, gc = goal

    if grid.is_obstacle(sr, sc) or grid.is_obstacle(gr, gc):
        return AStarResult(path=[], cost=SENTINEL, time_ms=0.0, nodes_expanded=0)

    t0 = time.perf_counter()

    g_score: dict[tuple[int, int], int] = {start: 0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    h0 = _heuristic(start, goal, grid.connectivity)
    # tie-break counter for stable heap ordering
    counter = 0
    heap: list[tuple[int, int, tuple[int, int]]] = [(h0, counter, start)]
    expanded = 0

    while heap:
        f, _, u = heapq.heappop(heap)
        g_u = g_score.get(u, SENTINEL)
        if f - _heuristic(u, goal, grid.connectivity) > g_u:
            continue
        expanded += 1
        if u == goal:
            break
        for nr, nc in grid.get_neighbors(u[0], u[1]):
            w = grid.edge_weight(u[0], u[1], nr, nc)
            if w >= SENTINEL:
                continue
            ng = g_u + w
            v = (nr, nc)
            if ng < g_score.get(v, SENTINEL):
                g_score[v] = ng
                prev[v] = u
                counter += 1
                heapq.heappush(heap, (ng + _heuristic(v, goal, grid.connectivity), counter, v))

    elapsed = (time.perf_counter() - t0) * 1000.0

    if goal not in g_score:
        return AStarResult(path=[], cost=SENTINEL, time_ms=elapsed, nodes_expanded=expanded)

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()

    return AStarResult(path=path, cost=g_score[goal], time_ms=elapsed, nodes_expanded=expanded)
