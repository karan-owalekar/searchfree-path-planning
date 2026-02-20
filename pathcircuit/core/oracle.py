from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathcircuit.core.tropical import SENTINEL, tropical_add, minplus_mv
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

    # Root of the hierarchy tree (None for single-block grids).
    root: MergeNode | None = field(default=None, repr=False)


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
# Core min-plus expansion primitive
# ---------------------------------------------------------------------------

def _expand_single_node(node: MergeNode, d: dict[tuple[int, int], int]) -> None:
    """Expand distances in *d* through a single node's D_global table.

    Pure algebraic: d_new[k] = min( d[k], min_b( d[bridge_b] + D_global[bridge_b, k] ) )
    Uses JIT-compiled kernel when available.
    """
    if node.D_global_cells is None:
        return

    cells = node.D_global_cells
    n = len(cells)

    bridge_idx: list[int] = []
    bridge_d: list[int] = []
    for i, c in enumerate(cells):
        if c in d:
            bridge_idx.append(i)
            bridge_d.append(d[c])

    if not bridge_idx:
        return

    bi = np.array(bridge_idx, dtype=np.int64)
    bd = np.array(bridge_d, dtype=np.int64)
    d_new = minplus_mv(node.D_global, bi, bd, n)

    for k in range(n):
        v = int(d_new[k])
        if v < d.get(cells[k], SENTINEL):
            d[cells[k]] = v


# ---------------------------------------------------------------------------
# Search-free hierarchical expansion (chain-based, kept for goal_exp)
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
        _expand_single_node(node, d)
    return d


# ---------------------------------------------------------------------------
# Top-down tree walk for full precomputation
# ---------------------------------------------------------------------------

def _walk_and_expand(node: BlockLike, d: dict[tuple[int, int], int]) -> None:
    """DFS top-down: expand *d* at this node, then recurse to children.

    Each hierarchy node is visited exactly once, eliminating redundant work
    that per-block reverse expansions would duplicate.
    """
    if isinstance(node, BaseBlock):
        return

    assert isinstance(node, MergeNode)
    _expand_single_node(node, d)

    if node.h_merge_left is not None:
        _expand_single_node(node.h_merge_left, d)
        _expand_single_node(node.h_merge_right, d)
        for child in (node.h_merge_left.child_left, node.h_merge_left.child_right,
                      node.h_merge_right.child_left, node.h_merge_right.child_right):
            _walk_and_expand(child, d)
    else:
        for child in (node.child_left, node.child_right):
            _walk_and_expand(child, d)


_BI = np.array(BOUNDARY_INDICES, dtype=np.int32)


def precompute_all_block_dists(
    oracle: Oracle,
    goal_exp: dict[tuple[int, int], int],
    goal: tuple[int, int],
) -> dict[tuple[int, int], np.ndarray]:
    """Precompute d(cell, goal) for every cell in every base block.

    Single top-down pass through the hierarchy tree propagates goal distances
    to all boundary cells, then each block's intra-block APSP matrix converts
    boundary distances to per-cell distances.

    Returns { (block_row, block_col): int64[16] }.
    """
    d = dict(goal_exp)

    if oracle.root is not None:
        _walk_and_expand(oracle.root, d)

    gr, gc = goal
    g_block = (gr // 4, gc // 4)
    g_local = (gr % 4) * 4 + (gc % 4)

    result: dict[tuple[int, int], np.ndarray] = {}
    for br, row in enumerate(oracle.base_blocks):
        for bc, bb in enumerate(row):
            btg = np.empty(12, dtype=np.int64)
            for b in range(12):
                btg[b] = d.get(bb.boundary_cells[b], SENTINEL)

            D_to_b = bb.D[:, _BI].astype(np.int64)
            sums = D_to_b + btg[np.newaxis, :]
            np.clip(sums, 0, SENTINEL, out=sums)
            dists = np.min(sums, axis=1)

            if (br, bc) == g_block:
                dists = np.minimum(dists, bb.D[:, g_local].astype(np.int64))

            result[(br, bc)] = dists

    return result


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
        root_node: MergeNode | None = root
    else:
        chains = {}
        root_node = None

    return Oracle(
        base_blocks=base_blocks,
        N=N,
        grid=grid,
        ancestor_chains=chains,
        root=root_node,
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
    block_dists: dict[tuple[int, int], np.ndarray] | None = None,
) -> int:
    if s == g:
        return 0

    # Fast path: precomputed block dists give O(1) lookup
    if block_dists is not None:
        sr, sc = s
        bk = (sr // 4, sc // 4)
        bd = block_dists.get(bk)
        if bd is not None:
            return int(bd[(sr % 4) * 4 + (sc % 4)])
        return SENTINEL

    sr, sc = s
    gr, gc = g
    s_br, s_bc = sr // 4, sc // 4

    bb_s = oracle.base_blocks[s_br][s_bc]
    s_local = (sr % 4) * 4 + (sc % 4)

    if s_br == gr // 4 and s_bc == gc // 4:
        g_local = (gr % 4) * 4 + (gc % 4)
        best = int(bb_s.D[s_local, g_local])
    else:
        best = SENTINEL

    initial_s: dict[tuple[int, int], int] = {}
    for a in range(12):
        d = int(bb_s.D[s_local, BOUNDARY_INDICES[a]])
        if d < SENTINEL:
            initial_s[bb_s.boundary_cells[a]] = d

    chain_s = oracle.ancestor_chains.get((s_br, s_bc), [])
    d_from_s = _expand_through_hierarchy(chain_s, initial_s)

    if goal_expansion is None:
        goal_expansion = precompute_goal_expansion(oracle, g)

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
    goal_exp: dict[tuple[int, int], int] | None = None,
    block_dists: dict[tuple[int, int], np.ndarray] | None = None,
) -> list[tuple[int, int]]:
    if s == g:
        return [s]

    if block_dists is not None:
        d_sg = query_distance(oracle, s, g, block_dists=block_dists)
    else:
        if goal_exp is None:
            goal_exp = precompute_goal_expansion(oracle, g)
        d_sg = query_distance(oracle, s, g, goal_exp)

    if d_sg >= SENTINEL:
        return []

    grid = oracle.grid
    path = [s]
    current = s

    if block_dists is not None:
        # Pure-lookup reconstruction: every d(neighbor, goal) is an array read
        while current != g:
            best_next = None
            best_cost = SENTINEL

            for nr, nc in grid.get_neighbors(current[0], current[1]):
                w = grid.edge_weight(current[0], current[1], nr, nc)
                if w >= SENTINEL:
                    continue

                bk = (nr // 4, nc // 4)
                bd = block_dists.get(bk)
                if bd is None:
                    continue
                d_ng = int(bd[(nr % 4) * 4 + (nc % 4)])
                total = tropical_add(w, d_ng)
                if total < best_cost:
                    best_cost = total
                    best_next = (nr, nc)

            if best_next is None:
                break
            path.append(best_next)
            current = best_next
    else:
        # Legacy lazy-cache path (fallback when block_dists not provided)
        if goal_exp is None:
            goal_exp = precompute_goal_expansion(oracle, g)

        block_cache: dict[tuple[int, int], np.ndarray] = {}

        while current != g:
            best_next = None
            best_cost = SENTINEL

            for nr, nc in grid.get_neighbors(current[0], current[1]):
                w = grid.edge_weight(current[0], current[1], nr, nc)
                if w >= SENTINEL:
                    continue

                bk = (nr // 4, nc // 4)
                if bk not in block_cache:
                    block_cache[bk] = _block_goal_dists(oracle, bk[0], bk[1], goal_exp, g)

                d_ng = int(block_cache[bk][(nr % 4) * 4 + (nc % 4)])
                total = tropical_add(w, d_ng)
                if total < best_cost:
                    best_cost = total
                    best_next = (nr, nc)

            if best_next is None:
                break
            path.append(best_next)
            current = best_next

    return path


# ---------------------------------------------------------------------------
# Legacy per-block reverse expansion (kept for fallback path)
# ---------------------------------------------------------------------------

def _block_goal_dists(
    oracle: Oracle,
    br: int,
    bc: int,
    goal_exp: dict[tuple[int, int], int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Return a 16-element int64 array: d(cell, goal) for every cell in block (br, bc)."""
    bb = oracle.base_blocks[br][bc]
    chain = oracle.ancestor_chains.get((br, bc), [])

    reverse_d = _expand_through_hierarchy(list(reversed(chain)), goal_exp) if chain else goal_exp

    btg = np.empty(12, dtype=np.int64)
    for b in range(12):
        btg[b] = reverse_d.get(bb.boundary_cells[b], SENTINEL)

    D_to_b = bb.D[:, _BI].astype(np.int64)
    sums = D_to_b + btg[np.newaxis, :]
    np.clip(sums, 0, SENTINEL, out=sums)
    dists = np.min(sums, axis=1)

    gr, gc = goal
    if gr // 4 == br and gc // 4 == bc:
        g_local = (gr % 4) * 4 + (gc % 4)
        dists = np.minimum(dists, bb.D[:, g_local].astype(np.int64))

    return dists
