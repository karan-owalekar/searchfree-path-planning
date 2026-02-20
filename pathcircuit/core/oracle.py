from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathcircuit.core.tropical import SENTINEL, tropical_add
from pathcircuit.core.base_block import BaseBlock, BOUNDARY_INDICES
from pathcircuit.core.hierarchy import MergeNode, BlockLike
from pathcircuit.grid import Grid


@dataclass
class Oracle:
    base_blocks: list[list[BaseBlock]]
    N: int
    grid: Grid

    # For each base block (br, bc): ordered list of ancestor MergeNodes
    # from the immediate parent up to the root.
    ancestor_chains: dict[tuple[int, int], list[MergeNode]] = field(
        default_factory=dict, repr=False
    )


# ---------------------------------------------------------------------------
# Ancestor-chain construction
# ---------------------------------------------------------------------------

def _build_ancestor_chains(root: MergeNode) -> dict[tuple[int, int], list[MergeNode]]:
    """Walk the hierarchy tree and record, for every base block, the ordered
    list of merge nodes from leaf (immediate parent) up to root."""
    chains: dict[tuple[int, int], list[MergeNode]] = {}

    def walk(node: BlockLike, suffix: list[MergeNode]) -> None:
        if isinstance(node, BaseBlock):
            chains[(node.block_row, node.block_col)] = list(suffix)
            return

        assert isinstance(node, MergeNode)
        if node.h_merge_left is not None:
            for h_merge in (node.h_merge_left, node.h_merge_right):
                child_suffix = [h_merge, node] + suffix
                for child in (h_merge.child_left, h_merge.child_right):
                    walk(child, child_suffix)
        else:
            child_suffix = [node] + suffix
            for child in (node.child_left, node.child_right):
                walk(child, child_suffix)

    walk(root, [])
    return chains


# ---------------------------------------------------------------------------
# Search-free hierarchical expansion
# ---------------------------------------------------------------------------

def _expand_through_hierarchy(
    chain: list[MergeNode],
    initial: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """Project *initial* distances upward through every node in *chain*
    using min-plus matrix-vector products on the stored D_global tables.

    Pure algebraic computation -- no search involved.
    """
    d = dict(initial)

    for node in chain:
        if node.D_global_cells is None:
            continue

        cells = node.D_global_cells
        n = len(cells)

        # Collect bridge cells: those already reachable AND present in this
        # node's D_global.
        bridge_idx: list[int] = []
        bridge_d: list[int] = []
        for i, c in enumerate(cells):
            if c in d:
                bridge_idx.append(i)
                bridge_d.append(d[c])

        if not bridge_idx:
            continue

        # Vectorised min-plus: d_new[k] = min_b (bridge_d[b] + D_global[b, k])
        bi = np.array(bridge_idx)
        bd = np.array(bridge_d, dtype=np.int64)
        D_sub = node.D_global[bi, :].astype(np.int64)       # (|bridges|, n)
        sums = bd[:, np.newaxis] + D_sub                     # broadcast
        np.clip(sums, 0, SENTINEL, out=sums)
        d_new = np.min(sums, axis=0)                         # (n,)

        for k in range(n):
            v = int(d_new[k])
            if v < d.get(cells[k], SENTINEL):
                d[cells[k]] = v

    return d


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def build_oracle(
    root: MergeNode | BaseBlock,
    base_blocks: list[list[BaseBlock]],
    N: int,
    grid: Grid,
) -> Oracle:
    if isinstance(root, MergeNode):
        chains = _build_ancestor_chains(root)
    else:
        chains = {}

    return Oracle(
        base_blocks=base_blocks,
        N=N,
        grid=grid,
        ancestor_chains=chains,
    )


# ---------------------------------------------------------------------------
# Goal-side expansion (precomputed once per Find-Path click)
# ---------------------------------------------------------------------------

def precompute_goal_expansion(
    oracle: Oracle, goal: tuple[int, int],
) -> dict[tuple[int, int], int]:
    """Expand distances from *goal*'s boundary cells up through the hierarchy.

    Returns {relay_cell: distance_to_goal} -- purely algebraic, no search.
    """
    gr, gc = goal
    bb_g = oracle.base_blocks[gr // 4][gc // 4]
    g_local = (gr % 4) * 4 + (gc % 4)

    initial: dict[tuple[int, int], int] = {}
    for b in range(12):
        d = int(bb_g.D[BOUNDARY_INDICES[b], g_local])
        if d < SENTINEL:
            initial[bb_g.boundary_cells[b]] = d

    chain = oracle.ancestor_chains.get((gr // 4, gc // 4), [])
    return _expand_through_hierarchy(chain, initial)


# ---------------------------------------------------------------------------
# Distance queries
# ---------------------------------------------------------------------------

def query_distance(
    oracle: Oracle,
    s: tuple[int, int],
    g: tuple[int, int],
    goal_expansion: dict[tuple[int, int], int] | None = None,
) -> int:
    if s == g:
        return 0

    sr, sc = s
    gr, gc = g
    s_br, s_bc = sr // 4, sc // 4

    bb_s = oracle.base_blocks[s_br][s_bc]
    s_local = (sr % 4) * 4 + (sc % 4)

    # Same base block: start with the local (intra-block) distance
    if s_br == gr // 4 and s_bc == gc // 4:
        g_local = (gr % 4) * 4 + (gc % 4)
        best = int(bb_s.D[s_local, g_local])
    else:
        best = SENTINEL

    # Source-side: local distances to boundary cells
    initial_s: dict[tuple[int, int], int] = {}
    for a in range(12):
        d = int(bb_s.D[s_local, BOUNDARY_INDICES[a]])
        if d < SENTINEL:
            initial_s[bb_s.boundary_cells[a]] = d

    # Expand source upward through the hierarchy (search-free)
    chain_s = oracle.ancestor_chains.get((s_br, s_bc), [])
    d_from_s = _expand_through_hierarchy(chain_s, initial_s)

    # Goal-side expansion (precomputed or on-demand)
    if goal_expansion is None:
        goal_expansion = precompute_goal_expansion(oracle, g)

    # Combine: d(s, g) = min_c (d_from_s[c] + d_to_g[c])
    for cell, ds in d_from_s.items():
        dg = goal_expansion.get(cell, SENTINEL)
        if ds < SENTINEL and dg < SENTINEL:
            total = ds + dg
            if total < best:
                best = total

    return min(best, SENTINEL)


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def reconstruct_path(
    oracle: Oracle,
    s: tuple[int, int],
    g: tuple[int, int],
) -> list[tuple[int, int]]:
    if s == g:
        return [s]

    goal_exp = precompute_goal_expansion(oracle, g)
    d_sg = query_distance(oracle, s, g, goal_exp)
    if d_sg >= SENTINEL:
        return []

    grid = oracle.grid
    path = [s]
    current = s

    while current != g:
        best_next = None
        best_cost = SENTINEL

        for nr, nc in grid.get_neighbors(current[0], current[1]):
            w = grid.edge_weight(current[0], current[1], nr, nc)
            if w >= SENTINEL:
                continue
            d_ng = query_distance(oracle, (nr, nc), g, goal_exp)
            total = tropical_add(w, d_ng)
            if total < best_cost:
                best_cost = total
                best_next = (nr, nc)

        if best_next is None:
            break

        path.append(best_next)
        current = best_next

    return path
