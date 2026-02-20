from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from pathcircuit.grid import Grid, SENTINEL


@dataclass
class DijkstraResult:
    path: list[tuple[int, int]]
    cost: int
    time_ms: float
    nodes_expanded: int


def dijkstra(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> DijkstraResult:
    sr, sc = start
    gr, gc = goal

    if grid.is_obstacle(sr, sc) or grid.is_obstacle(gr, gc):
        return DijkstraResult(path=[], cost=SENTINEL, time_ms=0.0, nodes_expanded=0)

    t0 = time.perf_counter()

    dist: dict[tuple[int, int], int] = {start: 0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[int, tuple[int, int]]] = [(0, start)]
    expanded = 0

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, SENTINEL):
            continue
        expanded += 1
        if u == goal:
            break
        for nr, nc in grid.get_neighbors(u[0], u[1]):
            w = grid.edge_weight(u[0], u[1], nr, nc)
            if w >= SENTINEL:
                continue
            nd = d + w
            v = (nr, nc)
            if nd < dist.get(v, SENTINEL):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    elapsed = (time.perf_counter() - t0) * 1000.0

    if goal not in dist:
        return DijkstraResult(path=[], cost=SENTINEL, time_ms=elapsed, nodes_expanded=expanded)

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()

    return DijkstraResult(path=path, cost=dist[goal], time_ms=elapsed, nodes_expanded=expanded)
