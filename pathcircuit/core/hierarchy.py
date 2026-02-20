from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from pathcircuit.core.tropical import (
    SENTINEL,
    tropical_matmul,
    tropical_matmul_with_witness,
)
from pathcircuit.core.base_block import BaseBlock
from pathcircuit.grid import Grid

BlockLike = Union[BaseBlock, "MergeNode"]


@dataclass
class MergeNode:
    level: int
    merge_type: str  # "horizontal" or "vertical"

    region_top: int
    region_left: int
    region_height: int
    region_width: int

    boundary_size: int
    boundary_cells: list[tuple[int, int]]
    boundary_cell_to_index: dict[tuple[int, int], int]

    T: np.ndarray

    interface_size: int
    interface_cells: list[tuple[int, int]]
    M_SS_star: np.ndarray
    M_ES_times_MSS_star: np.ndarray
    M_ES: np.ndarray
    M_SE: np.ndarray
    kleene_witnesses: list[np.ndarray]

    child_left: BlockLike
    child_right: BlockLike

    # Horizontal sub-merges stored on the vertical merge node
    h_merge_left: MergeNode | None = field(default=None, repr=False)
    h_merge_right: MergeNode | None = field(default=None, repr=False)

    # Filled during un-condensation
    D_global: np.ndarray | None = field(default=None, repr=False)
    D_global_cells: list[tuple[int, int]] | None = field(default=None, repr=False)
    D_global_cell_to_index: dict[tuple[int, int], int] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def enumerate_boundary(r_top: int, r_bot: int, c_left: int, c_right: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c_left, c_right + 1):
        cells.append((r_top, c))
    for r in range(r_top + 1, r_bot):
        cells.append((r, c_right))
    for c in range(c_left, c_right + 1):
        cells.append((r_bot, c))
    for r in range(r_top + 1, r_bot):
        cells.append((r, c_left))
    return cells


def are_grid_adjacent(c1: tuple[int, int], c2: tuple[int, int], connectivity: int) -> bool:
    if connectivity == 4:
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) == 1
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1])) == 1 and c1 != c2


def _edge_weight(c1: tuple[int, int], c2: tuple[int, int], grid: Grid) -> int:
    ci = int(grid.costs[c1[0], c1[1]])
    cj = int(grid.costs[c2[0], c2[1]])
    if ci >= SENTINEL or cj >= SENTINEL:
        return SENTINEL
    return ci + cj


# ---------------------------------------------------------------------------
# Kleene closure via repeated squaring
# ---------------------------------------------------------------------------

def kleene_closure(M_SS: np.ndarray, s_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    num_sq = max(1, math.ceil(math.log2(s_size))) if s_size > 1 else 1
    R = M_SS.copy()
    witnesses: list[np.ndarray] = []
    for _ in range(num_sq):
        R_new, W_t = tropical_matmul_with_witness(R, R)
        R = R_new
        witnesses.append(W_t)
    return R, witnesses


# ---------------------------------------------------------------------------
# Partition identification
# ---------------------------------------------------------------------------

def _identify_horizontal_partition(A: BlockLike, B: BlockLike):
    shared_col_A = A.region_left + A.region_width - 1
    shared_col_B = B.region_left
    r_top = A.region_top
    r_bot = A.region_top + A.region_height - 1

    I_A = [(r, shared_col_A) for r in range(r_top + 1, r_bot)]
    I_B = [(r, shared_col_B) for r in range(r_top + 1, r_bot)]
    S = I_A + I_B

    set_IA = set(I_A)
    set_IB = set(I_B)
    E_A = [c for c in A.boundary_cells if c not in set_IA]
    E_B = [c for c in B.boundary_cells if c not in set_IB]

    c_left = A.region_left
    c_right = B.region_left + B.region_width - 1
    C_boundary = enumerate_boundary(r_top, r_bot, c_left, c_right)

    return C_boundary, S, I_A, I_B, E_A, E_B


def _identify_vertical_partition(A: BlockLike, B: BlockLike):
    shared_row_A = A.region_top + A.region_height - 1
    shared_row_B = B.region_top
    c_left = A.region_left
    c_right = A.region_left + A.region_width - 1

    I_A = [(shared_row_A, c) for c in range(c_left + 1, c_right)]
    I_B = [(shared_row_B, c) for c in range(c_left + 1, c_right)]
    S = I_A + I_B

    set_IA = set(I_A)
    set_IB = set(I_B)
    E_A = [c for c in A.boundary_cells if c not in set_IA]
    E_B = [c for c in B.boundary_cells if c not in set_IB]

    r_top = A.region_top
    r_bot = B.region_top + B.region_height - 1
    C_boundary = enumerate_boundary(r_top, r_bot, c_left, c_right)

    return C_boundary, S, I_A, I_B, E_A, E_B


# ---------------------------------------------------------------------------
# Joint matrix assembly
# ---------------------------------------------------------------------------

def _assemble_joint_matrix(
    T_A: np.ndarray,
    T_B: np.ndarray,
    C_boundary: list[tuple[int, int]],
    S: list[tuple[int, int]],
    I_A: list[tuple[int, int]],
    I_B: list[tuple[int, int]],
    A: BlockLike,
    B: BlockLike,
    grid: Grid,
) -> np.ndarray:
    nE = len(C_boundary)
    nS = len(S)
    n = nE + nS

    # Build source mappings for E cells
    E_map: list[tuple[str, int]] = []
    for cell in C_boundary:
        if cell in A.boundary_cell_to_index:
            E_map.append(("A", A.boundary_cell_to_index[cell]))
        elif cell in B.boundary_cell_to_index:
            E_map.append(("B", B.boundary_cell_to_index[cell]))
        else:
            raise RuntimeError(f"Cell {cell} not in either boundary")

    S_A_map = [A.boundary_cell_to_index[c] for c in I_A]
    S_B_map = [B.boundary_cell_to_index[c] for c in I_B]
    nSA = len(S_A_map)

    M = np.full((n, n), SENTINEL, dtype=np.int32)

    # M_EE
    for i in range(nE):
        for j in range(nE):
            si, idx_i = E_map[i]
            sj, idx_j = E_map[j]
            if si == sj:
                T = T_A if si == "A" else T_B
                M[i, j] = T[idx_i, idx_j]
            elif are_grid_adjacent(C_boundary[i], C_boundary[j], grid.connectivity):
                M[i, j] = _edge_weight(C_boundary[i], C_boundary[j], grid)

    # M_ES / M_SE
    for i in range(nE):
        si, idx_i = E_map[i]
        for j in range(nS):
            sj_idx = nE + j
            if j < nSA:
                # S cell from I_A
                if si == "A":
                    val = int(T_A[idx_i, S_A_map[j]])
                    M[i, sj_idx] = val
                    M[sj_idx, i] = val
                elif are_grid_adjacent(C_boundary[i], I_A[j], grid.connectivity):
                    w = _edge_weight(C_boundary[i], I_A[j], grid)
                    if w < M[i, sj_idx]:
                        M[i, sj_idx] = w
                        M[sj_idx, i] = w
            else:
                # S cell from I_B
                jB = j - nSA
                if si == "B":
                    val = int(T_B[idx_i, S_B_map[jB]])
                    M[i, sj_idx] = val
                    M[sj_idx, i] = val
                elif are_grid_adjacent(C_boundary[i], I_B[jB], grid.connectivity):
                    w = _edge_weight(C_boundary[i], I_B[jB], grid)
                    if w < M[i, sj_idx]:
                        M[i, sj_idx] = w
                        M[sj_idx, i] = w

    # M_SS
    for i in range(nS):
        for j in range(nS):
            si = nE + i
            sj = nE + j
            inA_i = i < nSA
            inA_j = j < nSA

            if inA_i and inA_j:
                M[si, sj] = T_A[S_A_map[i], S_A_map[j]]
            elif (not inA_i) and (not inA_j):
                M[si, sj] = T_B[S_B_map[i - nSA], S_B_map[j - nSA]]
            else:
                if inA_i:
                    cell_a, cell_b = I_A[i], I_B[j - nSA]
                else:
                    cell_a, cell_b = I_A[j], I_B[i - nSA]
                if are_grid_adjacent(cell_a, cell_b, grid.connectivity):
                    w = _edge_weight(cell_a, cell_b, grid)
                    M[si, sj] = w
                    M[sj, si] = w

    return M


# ---------------------------------------------------------------------------
# Merge operators
# ---------------------------------------------------------------------------

def _merge(
    A: BlockLike,
    B: BlockLike,
    level: int,
    grid: Grid,
    merge_type: str,
) -> MergeNode:
    if merge_type == "horizontal":
        C_boundary, S, I_A, I_B, E_A, E_B = _identify_horizontal_partition(A, B)
    else:
        C_boundary, S, I_A, I_B, E_A, E_B = _identify_vertical_partition(A, B)

    nE = len(C_boundary)
    nS = len(S)

    M = _assemble_joint_matrix(A.T, B.T, C_boundary, S, I_A, I_B, A, B, grid)

    M_EE = M[:nE, :nE].copy()
    M_ES = M[:nE, nE:nE + nS].copy()
    M_SE = M[nE:nE + nS, :nE].copy()
    M_SS = M[nE:nE + nS, nE:nE + nS].copy()

    if nS > 0:
        M_SS_star, witnesses = kleene_closure(M_SS, nS)
        M_ES_times_MSS_star = tropical_matmul(M_ES, M_SS_star)
        schur_product = tropical_matmul(M_ES_times_MSS_star, M_SE)
        T_C = np.minimum(M_EE, schur_product)
    else:
        M_SS_star = np.empty((0, 0), dtype=np.int32)
        witnesses = []
        M_ES_times_MSS_star = np.empty((nE, 0), dtype=np.int32)
        T_C = M_EE.copy()

    if merge_type == "horizontal":
        rh = A.region_height
        rw = A.region_width + B.region_width
        rt = A.region_top
        rl = A.region_left
    else:
        rh = A.region_height + B.region_height
        rw = A.region_width
        rt = A.region_top
        rl = A.region_left

    return MergeNode(
        level=level,
        merge_type=merge_type,
        region_top=rt,
        region_left=rl,
        region_height=rh,
        region_width=rw,
        boundary_size=nE,
        boundary_cells=C_boundary,
        boundary_cell_to_index={c: i for i, c in enumerate(C_boundary)},
        T=T_C,
        interface_size=nS,
        interface_cells=S,
        M_SS_star=M_SS_star,
        M_ES_times_MSS_star=M_ES_times_MSS_star,
        M_ES=M_ES,
        M_SE=M_SE,
        kleene_witnesses=witnesses,
        child_left=A,
        child_right=B,
    )


def horizontal_merge(A: BlockLike, B: BlockLike, level: int, grid: Grid) -> MergeNode:
    return _merge(A, B, level, grid, "horizontal")


def vertical_merge(A: BlockLike, B: BlockLike, level: int, grid: Grid) -> MergeNode:
    return _merge(A, B, level, grid, "vertical")


# ---------------------------------------------------------------------------
# Full hierarchy construction
# ---------------------------------------------------------------------------

def build_hierarchy(
    base_blocks: list[list[BaseBlock]],
    N: int,
    grid: Grid,
) -> tuple[MergeNode, list[MergeNode]]:
    L = int(math.log2(N / 4))
    prev_level: list[list[BlockLike]] = [list(row) for row in base_blocks]
    all_merge_nodes: list[MergeNode] = []

    for level in range(L):
        rows_prev = len(prev_level)
        cols_prev = len(prev_level[0])
        rows_next = rows_prev // 2
        cols_next = cols_prev // 2

        next_level: list[list[BlockLike]] = []
        for gr in range(rows_next):
            row_blocks: list[BlockLike] = []
            for gc in range(cols_next):
                a = prev_level[2 * gr][2 * gc]
                b = prev_level[2 * gr][2 * gc + 1]
                c = prev_level[2 * gr + 1][2 * gc]
                d = prev_level[2 * gr + 1][2 * gc + 1]

                ab = horizontal_merge(a, b, level, grid)
                cd = horizontal_merge(c, d, level, grid)
                abcd = vertical_merge(ab, cd, level, grid)

                abcd.h_merge_left = ab
                abcd.h_merge_right = cd

                row_blocks.append(abcd)
                all_merge_nodes.extend([abcd, ab, cd])
            next_level.append(row_blocks)
        prev_level = next_level

    root = prev_level[0][0]
    assert isinstance(root, MergeNode)
    return root, all_merge_nodes
