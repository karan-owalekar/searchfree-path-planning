from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathcircuit.core.tropical import (
    SENTINEL,
    tropical_matmul_with_witness,
)
from pathcircuit.grid import Grid

# -- 4x4 block layout constants --

BOUNDARY_INDICES = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
INTERIOR_INDICES = [5, 6, 9, 10]
LOCAL_TO_BOUNDARY = {v: i for i, v in enumerate(BOUNDARY_INDICES)}

NEIGHBORS_4 = {
    0: [1, 4],
    1: [0, 2, 5],
    2: [1, 3, 6],
    3: [2, 7],
    4: [0, 5, 8],
    5: [1, 4, 6, 9],
    6: [2, 5, 7, 10],
    7: [3, 6, 11],
    8: [4, 9, 12],
    9: [5, 8, 10, 13],
    10: [6, 9, 11, 14],
    11: [7, 10, 15],
    12: [8, 13],
    13: [9, 12, 14],
    14: [10, 13, 15],
    15: [11, 14],
}

# 8-connectivity adds diagonal neighbors within a 4x4 block
NEIGHBORS_8: dict[int, list[int]] = {}
for _cell in range(16):
    _r, _c = divmod(_cell, 4)
    _nbrs = list(NEIGHBORS_4[_cell])
    for _dr, _dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < 4 and 0 <= _nc < 4:
            _nbrs.append(_nr * 4 + _nc)
    NEIGHBORS_8[_cell] = _nbrs


@dataclass
class BaseBlock:
    block_row: int
    block_col: int
    D: np.ndarray                   # 16x16 all-pairs shortest paths
    T: np.ndarray                   # 12x12 transfer matrix
    witnesses: list[np.ndarray]     # 4 witness matrices, each 16x16
    boundary_cells: list[tuple[int, int]]
    boundary_cell_to_index: dict[tuple[int, int], int]
    region_top: int
    region_left: int
    region_height: int = 4
    region_width: int = 4

    # Set during un-condensation
    D_global: np.ndarray | None = field(default=None, repr=False)
    D_global_cells: list[tuple[int, int]] | None = field(default=None, repr=False)
    D_global_cell_to_index: dict[tuple[int, int], int] | None = field(default=None, repr=False)


def build_weight_matrix(br: int, bc: int, grid: Grid) -> np.ndarray:
    """Build the 16x16 weighted adjacency matrix for a 4x4 base block."""
    W = np.full((16, 16), SENTINEL, dtype=np.int32)

    neighbors = NEIGHBORS_8 if grid.connectivity == 8 else NEIGHBORS_4

    for i in range(16):
        W[i, i] = 0
        ri = br * 4 + i // 4
        ci = bc * 4 + i % 4
        cost_i = int(grid.costs[ri, ci])

        for j in neighbors[i]:
            rj = br * 4 + j // 4
            cj = bc * 4 + j % 4
            cost_j = int(grid.costs[rj, cj])

            if cost_i >= SENTINEL or cost_j >= SENTINEL:
                W[i, j] = SENTINEL
            else:
                W[i, j] = cost_i + cost_j

    return W


def compute_base_block(br: int, bc: int, grid: Grid) -> BaseBlock:
    """Compute the base block at grid position (br, bc)."""
    W = build_weight_matrix(br, bc, grid)

    S1, w1 = tropical_matmul_with_witness(W, W)
    S2, w2 = tropical_matmul_with_witness(S1, S1)
    S3, w3 = tropical_matmul_with_witness(S2, S2)
    D, w4 = tropical_matmul_with_witness(S3, S3)

    T = np.empty((12, 12), dtype=np.int32)
    for a in range(12):
        for b in range(12):
            T[a, b] = D[BOUNDARY_INDICES[a], BOUNDARY_INDICES[b]]

    boundary_cells: list[tuple[int, int]] = []
    boundary_cell_to_index: dict[tuple[int, int], int] = {}
    for bi, local in enumerate(BOUNDARY_INDICES):
        r = br * 4 + local // 4
        c = bc * 4 + local % 4
        boundary_cells.append((r, c))
        boundary_cell_to_index[(r, c)] = bi

    return BaseBlock(
        block_row=br,
        block_col=bc,
        D=D,
        T=T,
        witnesses=[w1, w2, w3, w4],
        boundary_cells=boundary_cells,
        boundary_cell_to_index=boundary_cell_to_index,
        region_top=br * 4,
        region_left=bc * 4,
    )
