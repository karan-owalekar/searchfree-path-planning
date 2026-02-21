"""Eagerly compile all JIT / GPU kernels so the first real query is fast.

With cache=True on every @njit, compilation results are persisted to disk.
First-ever launch compiles (~2 s); subsequent launches load from cache (~0.1 s).
"""
from __future__ import annotations

import numpy as np


def warmup_jit() -> None:
    from pathcircuit.grid import Grid
    from pathcircuit.core.pipeline import build_pathcircuit_oracle
    from pathcircuit.core.oracle import query_distance, reconstruct_path
    from pathcircuit.compare.dijkstra import dijkstra
    from pathcircuit.compare.astar import astar

    for conn in (4, 8):
        g = Grid(8, 8, connectivity=conn)
        g.set_obstacle(3, 3)
        o = build_pathcircuit_oracle(g)
        query_distance(o, (1, 1), (6, 6))
        reconstruct_path(o, (1, 1), (6, 6))
        dijkstra(g, (1, 1), (6, 6))
        astar(g, (1, 1), (6, 6))


def warmup_gpu() -> None:
    """Trigger NVRTC compilation of all CUDA RawKernels on a tiny grid."""
    from pathcircuit.grid import Grid
    from pathcircuit.core.pipeline import build_pathcircuit_oracle
    from pathcircuit.gpu.oracle_gpu import gpu_reconstruct_path
    from pathcircuit.gpu.dijkstra_gpu import gpu_dijkstra

    for conn in (4, 8):
        g = Grid(8, 8, connectivity=conn)
        g.set_obstacle(3, 3)
        o = build_pathcircuit_oracle(g)
        gpu_reconstruct_path(o, (1, 1), (6, 6))
        gpu_dijkstra(g, (1, 1), (6, 6))
