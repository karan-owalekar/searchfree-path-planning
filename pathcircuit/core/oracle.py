from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathcircuit.core.tropical import SENTINEL, tropical_add
from pathcircuit.core.base_block import (
    BaseBlock, BOUNDARY_INDICES, LOCAL_TO_BOUNDARY,
    NEIGHBORS_4, NEIGHBORS_8,
)
from pathcircuit.core.hierarchy import MergeNode, BlockLike
from pathcircuit.grid import Grid

try:
    from numba import njit as _njit, prange as _prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

_BI = np.array(BOUNDARY_INDICES, dtype=np.int32)

_NBRS_4 = np.full((16, 8), -1, dtype=np.int32)
_NBRS_8 = np.full((16, 8), -1, dtype=np.int32)
for _c in range(16):
    for _j, _n in enumerate(NEIGHBORS_4[_c]):
        _NBRS_4[_c, _j] = _n
    for _j, _n in enumerate(NEIGHBORS_8[_c]):
        _NBRS_8[_c, _j] = _n

_LOCAL_TO_BI = np.full(16, -1, dtype=np.int32)
for _local, _bi in LOCAL_TO_BOUNDARY.items():
    _LOCAL_TO_BI[_local] = _bi


@dataclass
class Oracle:
    base_blocks: list[list[BaseBlock]]
    N: int
    grid: Grid

    # Pre-stacked intra-block APSP: shape (n_blocks, 16, 16)
    all_D: np.ndarray = field(repr=False)

    # Flat boundary-to-boundary global distance table.
    # boundary_dist[bid_a, bid_b] = shortest global distance between
    # boundary cell a and boundary cell b.
    # bid = (block_row * blocks_c + block_col) * 12 + boundary_index
    boundary_dist: np.ndarray = field(repr=False)

    # Per-block boundary IDs: shape (blocks_r, blocks_c, 12)
    block_bids: np.ndarray = field(repr=False)

    # Pre-flattened costs array for JIT kernels: int64[rows*cols]
    costs_flat: np.ndarray = field(repr=False)

    # --- Linked-list path retrieval structures ---

    # Intra-block path atlas: int8[n_blocks_r, n_blocks_c, 12, 12, 9]
    # atlas[br,bc,entry_bi,exit_bi,0] = path_length
    # atlas[br,bc,entry_bi,exit_bi,1..length] = local cell indices
    atlas: np.ndarray | None = field(default=None, repr=False)

    # Boundary successor table: int32[B, B]  (target-major)
    # successor[target_bid, current_bid] = next boundary ID toward target
    boundary_successor: np.ndarray | None = field(default=None, repr=False)

    # Boundary coord lookup: int32[B, 2]  (row, col)
    boundary_coords: np.ndarray | None = field(default=None, repr=False)

    # Per-boundary-cell local boundary index: int8[B]
    boundary_local_bi: np.ndarray | None = field(default=None, repr=False)


# ===================================================================
# BUILD-TIME HELPERS (hierarchy linearization — used only during build,
# not stored in the Oracle)
# ===================================================================

def _build_ancestor_chains(root: MergeNode) -> dict[tuple[int, int], list[MergeNode]]:
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


def _linearize_nodes(nodes: list[MergeNode], N: int) -> tuple:
    valid = [n for n in nodes if n.D_global_cells is not None]
    n_nodes = len(valid)
    if n_nodes == 0:
        z64 = np.empty(0, dtype=np.int64)
        z32 = np.empty(0, dtype=np.int32)
        return z32, z64, z64, z64, z32, 0

    sizes = np.array([len(n.D_global_cells) for n in valid], dtype=np.int32)
    cell_offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    cell_offsets[1:] = np.cumsum(sizes)

    cell_flat = np.empty(int(cell_offsets[-1]), dtype=np.int64)
    D_offsets = np.zeros(n_nodes + 1, dtype=np.int64)

    for i, node in enumerate(valid):
        co = int(cell_offsets[i])
        for j, (r, c) in enumerate(node.D_global_cells):
            cell_flat[co + j] = r * N + c
        D_offsets[i + 1] = D_offsets[i] + int(sizes[i]) * int(sizes[i])

    D_flat = np.empty(int(D_offsets[-1]), dtype=np.int32)
    for i, node in enumerate(valid):
        D_flat[int(D_offsets[i]):int(D_offsets[i + 1])] = node.D_global.ravel()

    return D_flat, D_offsets, cell_flat, cell_offsets, sizes, n_nodes


def _collect_dfs_order(node: BlockLike) -> list[MergeNode]:
    if isinstance(node, BaseBlock):
        return []
    assert isinstance(node, MergeNode)
    result: list[MergeNode] = [node]
    if node.h_merge_left is not None:
        result.append(node.h_merge_left)
        result.append(node.h_merge_right)
        for child in (node.h_merge_left.child_left, node.h_merge_left.child_right,
                      node.h_merge_right.child_left, node.h_merge_right.child_right):
            result.extend(_collect_dfs_order(child))
    else:
        for child in (node.child_left, node.child_right):
            result.extend(_collect_dfs_order(child))
    return result


# ===================================================================
# BUILD-TIME JIT: hierarchy expansion (used only to build the table)
# ===================================================================

if _HAS_NUMBA:
    @_njit(cache=True)
    def _jit_expand(d_arr, D_flat, D_offsets, cell_flat,  # type: ignore[misc]
                    cell_offsets, sizes, n_nodes):
        _S = np.int64(536_870_912)
        for ni in range(n_nodes):
            n = np.int64(sizes[ni])
            if n == 0:
                continue
            co = cell_offsets[ni]
            do = D_offsets[ni]

            n_br = np.int64(0)
            bi = np.empty(n, dtype=np.int64)
            bd = np.empty(n, dtype=np.int64)
            for i in range(n):
                cv = d_arr[cell_flat[co + i]]
                if cv < _S:
                    bi[n_br] = i
                    bd[n_br] = cv
                    n_br += 1

            if n_br == 0:
                continue

            for k in range(n):
                ck = cell_flat[co + k]
                best = d_arr[ck]
                for b in range(n_br):
                    dg = np.int64(D_flat[do + bi[b] * n + k])
                    if dg >= _S:
                        continue
                    val = bd[b] + dg
                    if val < best:
                        best = val
                d_arr[ck] = best

    @_njit(cache=True, nogil=True)
    def _process_block_boundary(  # type: ignore[misc]
        D_block, base_bid, N, boundary_flat, n_boundary, BI,
        chain_D, chain_Do, chain_cf, chain_co, chain_sz, chain_n,
        tree_D, tree_Do, tree_cf, tree_co, tree_sz, tree_n,
        out,
    ):
        """Process all 12 boundary cells of one block. Writes to out[12, n_boundary].

        nogil=True releases the GIL so multiple blocks run in parallel threads.
        """
        _S = np.int64(536_870_912)
        for b in range(12):
            d_arr = np.full(N * N, _S, dtype=np.int64)

            src_local = BI[b]
            for j in range(12):
                d_ij = np.int64(D_block[src_local, BI[j]])
                if d_ij < _S:
                    d_arr[boundary_flat[base_bid + j]] = d_ij

            if chain_n > 0:
                _jit_expand(d_arr, chain_D, chain_Do, chain_cf,
                            chain_co, chain_sz, chain_n)
            if tree_n > 0:
                _jit_expand(d_arr, tree_D, tree_Do, tree_cf,
                            tree_co, tree_sz, tree_n)

            for bid2 in range(n_boundary):
                d = d_arr[boundary_flat[bid2]]
                if d < _S:
                    out[b, bid2] = np.int32(d)


# ===================================================================
# BUILD: flat boundary distance table
# ===================================================================

def _build_boundary_table(
    base_blocks: list[list[BaseBlock]],
    blocks_r: int,
    blocks_c: int,
    N: int,
    boundary_flat: np.ndarray,
    n_boundary: int,
    tree_data: tuple,
    chain_data: dict[tuple[int, int], tuple],
) -> np.ndarray:
    """Build the all-pairs boundary distance table.

    Dispatches blocks to parallel threads; each block's 12 boundary cells
    are expanded through the hierarchy independently.

    Returns int32[n_boundary, n_boundary].
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    boundary_dist = np.full((n_boundary, n_boundary), SENTINEL, dtype=np.int32)
    tree_D, tree_Do, tree_cf, tree_co, tree_sz, tree_n = tree_data

    empty_i32 = np.empty(0, dtype=np.int32)
    empty_i64 = np.empty(0, dtype=np.int64)

    tasks: list[tuple] = []
    for br in range(blocks_r):
        for bc in range(blocks_c):
            block_idx = br * blocks_c + bc
            base_bid = block_idx * 12
            D_block = base_blocks[br][bc].D

            g_key = (br, bc)
            if g_key in chain_data:
                cD, cDo, ccf, cco, csz, cn = chain_data[g_key]
            else:
                cD, cDo, ccf, cco, csz, cn = empty_i32, empty_i64, empty_i64, empty_i64, empty_i32, 0

            out = np.full((12, n_boundary), SENTINEL, dtype=np.int32)
            tasks.append((D_block, base_bid, cD, cDo, ccf, cco, csz, cn, out))

    n_blocks = len(tasks)

    if _HAS_NUMBA and n_blocks >= 4:
        n_workers = min(os.cpu_count() or 4, n_blocks)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for args in tasks:
                D_block, base_bid, cD, cDo, ccf, cco, csz, cn, out = args
                fut = pool.submit(
                    _process_block_boundary,
                    D_block, base_bid, N, boundary_flat, n_boundary, _BI,
                    cD, cDo, ccf, cco, csz, cn,
                    tree_D, tree_Do, tree_cf, tree_co, tree_sz, tree_n,
                    out,
                )
                futures[fut] = (base_bid, out)

            for fut in as_completed(futures):
                fut.result()
                base_bid, out = futures[fut]
                boundary_dist[base_bid:base_bid + 12] = np.minimum(
                    boundary_dist[base_bid:base_bid + 12], out)
    else:
        for args in tasks:
            D_block, base_bid, cD, cDo, ccf, cco, csz, cn, out = args
            if _HAS_NUMBA:
                _process_block_boundary(
                    D_block, base_bid, N, boundary_flat, n_boundary, _BI,
                    cD, cDo, ccf, cco, csz, cn,
                    tree_D, tree_Do, tree_cf, tree_co, tree_sz, tree_n,
                    out,
                )
            else:
                for b in range(12):
                    d_arr = np.full(N * N, SENTINEL, dtype=np.int64)
                    src_local = BOUNDARY_INDICES[b]
                    for j in range(12):
                        d_ij = int(D_block[src_local, BOUNDARY_INDICES[j]])
                        if d_ij < SENTINEL:
                            d_arr[boundary_flat[base_bid + j]] = d_ij
                    dists = d_arr[boundary_flat].astype(np.int32)
                    np.minimum(out[b], dists, out=out[b])

            boundary_dist[base_bid:base_bid + 12] = np.minimum(
                boundary_dist[base_bid:base_bid + 12], out)

    return boundary_dist


# ===================================================================
# LINKED-LIST BUILD: atlas, adjacency, successor
# ===================================================================

def _build_atlas(base_blocks, blocks_r, blocks_c, connectivity):
    """Build intra-block path atlas: int8[blocks_r, blocks_c, 12, 12, 9]."""
    nbrs_map = NEIGHBORS_8 if connectivity == 8 else NEIGHBORS_4
    atlas = np.zeros((blocks_r, blocks_c, 12, 12, 9), dtype=np.int8)

    for br in range(blocks_r):
        for bc in range(blocks_c):
            bb = base_blocks[br][bc]
            D = bb.D
            for ei in range(12):
                e_local = BOUNDARY_INDICES[ei]
                for xi in range(12):
                    x_local = BOUNDARY_INDICES[xi]
                    if ei == xi:
                        atlas[br, bc, ei, xi, 0] = 1
                        atlas[br, bc, ei, xi, 1] = e_local
                        continue
                    if int(D[e_local, x_local]) >= SENTINEL:
                        continue
                    path = [e_local]
                    cur = e_local
                    for _ in range(16):
                        if cur == x_local:
                            break
                        best_n, best_c = -1, SENTINEL
                        for n in nbrs_map[cur]:
                            w = int(D[cur, n]) - int(D[n, n])
                            if w >= SENTINEL:
                                w_raw = int(D[cur, cur]) + int(D[n, n])
                                if w_raw >= SENTINEL:
                                    continue
                                rem = int(D[n, x_local])
                                if rem >= SENTINEL:
                                    continue
                                total = w_raw + rem
                            else:
                                rem = int(D[n, x_local])
                                if rem >= SENTINEL:
                                    continue
                                total = w + rem
                            if total < best_c:
                                best_c = total
                                best_n = n
                        if best_n < 0:
                            break
                        path.append(best_n)
                        cur = best_n
                    if cur == x_local:
                        plen = min(len(path), 8)
                        atlas[br, bc, ei, xi, 0] = plen
                        for k in range(plen):
                            atlas[br, bc, ei, xi, k + 1] = path[k]
    return atlas


def _build_adjacency_flat(base_blocks, blocks_r, blocks_c, N, grid,
                          block_bids, boundary_dist):
    """Build boundary adjacency in flat CSR arrays for JIT."""
    _S = SENTINEL
    n_bnd = blocks_r * blocks_c * 12
    connectivity = grid.connectivity

    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_bnd)]

    for br in range(blocks_r):
        for bc in range(blocks_c):
            bb = base_blocks[br][bc]
            base_bid = (br * blocks_c + bc) * 12
            for ai in range(12):
                a_bid = base_bid + ai
                # Intra-block: other boundary cells via T matrix
                for bi in range(12):
                    if bi == ai:
                        continue
                    cost = int(bb.T[ai, bi])
                    if cost >= _S:
                        continue
                    b_bid = base_bid + bi
                    adj[a_bid].append((b_bid, cost))

    # Cross-block edges
    for br in range(blocks_r):
        for bc in range(blocks_c):
            bb = base_blocks[br][bc]
            base_bid = (br * blocks_c + bc) * 12
            for ai in range(12):
                a_bid = base_bid + ai
                ar, ac = bb.boundary_cells[ai]
                if connectivity == 8:
                    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                else:
                    offsets = [(-1,0),(0,-1),(0,1),(1,0)]
                for dr, dc in offsets:
                    nr, nc = ar + dr, ac + dc
                    if nr < 0 or nr >= N or nc < 0 or nc >= N:
                        continue
                    n_br, n_bc = nr // 4, nc // 4
                    if n_br == br and n_bc == bc:
                        continue
                    n_local = (nr % 4) * 4 + (nc % 4)
                    n_bi = LOCAL_TO_BOUNDARY.get(n_local, -1)
                    if n_bi < 0:
                        continue
                    n_bid = (n_br * blocks_c + n_bc) * 12 + n_bi
                    ci = int(grid.costs[ar, ac])
                    cj = int(grid.costs[nr, nc])
                    if ci >= _S or cj >= _S:
                        continue
                    adj[a_bid].append((n_bid, ci + cj))

    # Deduplicate: keep min cost per neighbor
    for a in range(n_bnd):
        best: dict[int, int] = {}
        for nbr, cost in adj[a]:
            if nbr not in best or cost < best[nbr]:
                best[nbr] = cost
        adj[a] = list(best.items())

    # Flatten to CSR
    total_edges = sum(len(a) for a in adj)
    adj_nbrs = np.empty(total_edges, dtype=np.int32)
    adj_costs = np.empty(total_edges, dtype=np.int32)
    adj_offsets = np.empty(n_bnd + 1, dtype=np.int32)
    idx = 0
    for a in range(n_bnd):
        adj_offsets[a] = idx
        for nbr, cost in adj[a]:
            adj_nbrs[idx] = nbr
            adj_costs[idx] = cost
            idx += 1
    adj_offsets[n_bnd] = idx

    return adj_nbrs, adj_costs, adj_offsets


if _HAS_NUMBA:
    @_njit(cache=True, parallel=True)
    def _build_successor_jit(boundary_dist, adj_nbrs, adj_costs, adj_offsets, B):  # type: ignore[misc]
        """Build B×B successor table. successor[target, current] = next hop."""
        _S = np.int32(536_870_912)
        successor = np.full((B, B), np.int32(-1), dtype=np.int32)
        for target in _prange(B):
            for current in range(B):
                if current == target:
                    successor[target, current] = target
                    continue
                d_ct = boundary_dist[current, target]
                if d_ct >= _S:
                    continue
                best_next = np.int32(-1)
                best_cost = np.int64(_S)
                start = adj_offsets[current]
                end = adj_offsets[current + 1]
                for idx in range(start, end):
                    nbr = adj_nbrs[idx]
                    ec = np.int64(adj_costs[idx])
                    d_nt = np.int64(boundary_dist[nbr, target])
                    if d_nt >= _S:
                        continue
                    total = ec + d_nt
                    if total < best_cost or (total == best_cost and nbr < best_next):
                        best_cost = total
                        best_next = nbr
                successor[target, current] = best_next
        return successor


def _build_linked_list_structures(base_blocks, blocks_r, blocks_c, N, grid,
                                  block_bids, boundary_dist):
    """Build atlas + adjacency + successor table."""
    atlas = _build_atlas(base_blocks, blocks_r, blocks_c, grid.connectivity)

    adj_nbrs, adj_costs, adj_offsets = _build_adjacency_flat(
        base_blocks, blocks_r, blocks_c, N, grid, block_bids, boundary_dist)

    B = blocks_r * blocks_c * 12
    if _HAS_NUMBA:
        successor = _build_successor_jit(boundary_dist, adj_nbrs, adj_costs,
                                         adj_offsets, B)
    else:
        successor = np.full((B, B), -1, dtype=np.int32)
        _S = SENTINEL
        for target in range(B):
            for current in range(B):
                if current == target:
                    successor[target, current] = current
                    continue
                d_ct = int(boundary_dist[current, target])
                if d_ct >= _S:
                    continue
                best_next, best_cost = -1, _S
                for idx in range(adj_offsets[current], adj_offsets[current + 1]):
                    nbr = int(adj_nbrs[idx])
                    ec = int(adj_costs[idx])
                    d_nt = int(boundary_dist[nbr, target])
                    if d_nt >= _S:
                        continue
                    total = ec + d_nt
                    if total < best_cost or (total == best_cost and nbr < best_next):
                        best_cost = total
                        best_next = nbr
                successor[target, current] = best_next

    # Boundary coordinate lookup
    boundary_coords = np.empty((B, 2), dtype=np.int32)
    boundary_local_bi = np.empty(B, dtype=np.int8)
    for br in range(blocks_r):
        for bc in range(blocks_c):
            bb = base_blocks[br][bc]
            base_bid = (br * blocks_c + bc) * 12
            for bi in range(12):
                bid = base_bid + bi
                r, c = bb.boundary_cells[bi]
                boundary_coords[bid] = (r, c)
                boundary_local_bi[bid] = bi

    return atlas, successor, boundary_coords, boundary_local_bi


# ===================================================================
# ORACLE CONSTRUCTION
# ===================================================================

def build_oracle(
    root: MergeNode | BaseBlock,
    base_blocks: list[list[BaseBlock]],
    N: int,
    grid: Grid,
    *,
    build_linked_list: bool = True,
) -> Oracle:
    blocks_r = len(base_blocks)
    blocks_c = len(base_blocks[0])
    n_boundary = blocks_r * blocks_c * 12

    all_D = np.empty((blocks_r * blocks_c, 16, 16), dtype=np.int32)
    for br, row in enumerate(base_blocks):
        for bc, bb in enumerate(row):
            all_D[br * blocks_c + bc] = bb.D

    block_bids = np.arange(n_boundary, dtype=np.int32).reshape(blocks_r, blocks_c, 12)

    boundary_flat = np.empty(n_boundary, dtype=np.int64)
    for br, row in enumerate(base_blocks):
        for bc, bb in enumerate(row):
            bid0 = (br * blocks_c + bc) * 12
            for b in range(12):
                r, c = bb.boundary_cells[b]
                boundary_flat[bid0 + b] = r * N + c

    if isinstance(root, MergeNode):
        dfs_nodes = _collect_dfs_order(root)
        tree_data = _linearize_nodes(dfs_nodes, N)

        chains = _build_ancestor_chains(root)
        chain_data = {k: _linearize_nodes(v, N) for k, v in chains.items()}

        boundary_dist = _build_boundary_table(
            base_blocks, blocks_r, blocks_c, N,
            boundary_flat, n_boundary, tree_data, chain_data,
        )
    else:
        bb = base_blocks[0][0]
        boundary_dist = np.full((12, 12), SENTINEL, dtype=np.int32)
        if bb.D_global is not None:
            np.minimum(boundary_dist, bb.D_global[:12, :12], out=boundary_dist)

    costs_flat = grid.costs.ravel().astype(np.int64)

    atlas = successor = boundary_coords = boundary_local_bi = None
    if build_linked_list:
        atlas, successor, boundary_coords, boundary_local_bi = \
            _build_linked_list_structures(
                base_blocks, blocks_r, blocks_c, N, grid,
                block_bids, boundary_dist)

    return Oracle(
        base_blocks=base_blocks, N=N, grid=grid,
        all_D=all_D, boundary_dist=boundary_dist, block_bids=block_bids,
        costs_flat=costs_flat,
        atlas=atlas,
        boundary_successor=successor,
        boundary_coords=boundary_coords,
        boundary_local_bi=boundary_local_bi,
    )


# ===================================================================
# QUERY-TIME JIT KERNELS (pure lookups, zero hierarchy traversal)
# ===================================================================

if _HAS_NUMBA:
    @_njit(cache=True)
    def _query_flat(D_s, D_g, s_local, g_local,  # type: ignore[misc]
                    s_bids, g_bids, boundary_dist, BI):
        """d(s,g) = min_{a,b}( D_s[s,a] + bd[a,b] + D_g[b,g] ).  144 lookups."""
        _S = np.int64(536_870_912)

        # Compact valid goal-boundary entries
        n_vg = np.int32(0)
        vg_d = np.empty(12, dtype=np.int64)
        vg_bid = np.empty(12, dtype=np.int32)
        for b in range(12):
            d = np.int64(D_g[BI[b], g_local])
            if d < _S:
                vg_d[n_vg] = d
                vg_bid[n_vg] = g_bids[b]
                n_vg += 1

        best = _S
        for a in range(12):
            d_sa = np.int64(D_s[s_local, BI[a]])
            if d_sa >= _S:
                continue
            ba = s_bids[a]
            for vb in range(n_vg):
                d_mid = np.int64(boundary_dist[ba, vg_bid[vb]])
                if d_mid >= _S:
                    continue
                total = d_sa + d_mid + vg_d[vb]
                if total < best:
                    best = total
        return best

    @_njit(cache=True, parallel=True)
    def _block_dists_flat(all_D, boundary_dist, block_bids,  # type: ignore[misc]
                          blocks_r, blocks_c, BI,
                          g_block_r, g_block_c, g_local, g_bids, D_g):
        """Precompute d(cell, goal) for all cells via pure table lookups + prange."""
        _S = np.int64(536_870_912)
        result = np.empty((blocks_r, blocks_c, 16), dtype=np.int64)

        for idx in _prange(blocks_r * blocks_c):
            br = idx // blocks_c
            bc = idx % blocks_c
            D = all_D[idx]
            s_bids = block_bids[br, bc]

            for cell in range(16):
                best = _S
                for a in range(12):
                    d_ca = np.int64(D[cell, BI[a]])
                    if d_ca >= _S:
                        continue
                    ba = s_bids[a]
                    for b in range(12):
                        d_bg = np.int64(D_g[BI[b], g_local])
                        if d_bg >= _S:
                            continue
                        d_mid = np.int64(boundary_dist[ba, g_bids[b]])
                        if d_mid >= _S:
                            continue
                        total = d_ca + d_mid + d_bg
                        if total < best:
                            best = total
                result[br, bc, cell] = best

            if br == g_block_r and bc == g_block_c:
                for cell in range(16):
                    ld = np.int64(D[cell, g_local])
                    if ld < result[br, bc, cell]:
                        result[br, bc, cell] = ld

        return result

    @_njit(cache=True)
    def _jit_reconstruct_inline(costs, rows, cols, connectivity,  # type: ignore[misc]
                                sr, sc, gr, gc,
                                all_D, boundary_dist, block_bids, BI,
                                g_br, g_bc, g_local):
        """Greedy walk with per-block cached distances.

        Per-block: fetches 144 mid-distances from the large boundary table
        once into a tiny L1-resident array, then reuses for all 16 cells.
        """
        _S = np.int64(536_870_912)
        blocks_r = rows // 4
        blocks_c = cols // 4
        max_len = rows * cols
        path_r = np.empty(max_len, dtype=np.int32)
        path_c = np.empty(max_len, dtype=np.int32)
        path_r[0] = sr
        path_c[0] = sc
        length = 1

        D_g = all_D[g_br * blocks_c + g_bc]
        g_bids = block_bids[g_br, g_bc]

        # Precompute goal-side boundary distances (fixed for entire walk)
        n_vg = np.int32(0)
        vg_idx = np.empty(12, dtype=np.int32)
        vg_d = np.empty(12, dtype=np.int64)
        vg_bid = np.empty(12, dtype=np.int32)
        for b in range(12):
            d = np.int64(D_g[BI[b], g_local])
            if d < _S:
                vg_idx[n_vg] = b
                vg_d[n_vg] = d
                vg_bid[n_vg] = g_bids[b]
                n_vg += 1

        if connectivity == 8:
            dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:
            dr = np.array([-1, 0, 0, 1], dtype=np.int32)
            dc = np.array([0, -1, 1, 0], dtype=np.int32)

        cache = np.full((blocks_r, blocks_c, 16), -np.int64(1), dtype=np.int64)
        mid = np.empty((12, 12), dtype=np.int64)

        n_nbrs = len(dr)
        cr, cc = sr, sc

        while cr != gr or cc != gc:
            best_nr = np.int32(-1)
            best_nc = np.int32(-1)
            best_cost = _S
            cu = np.int64(costs[cr * cols + cc])

            for k in range(n_nbrs):
                nr = cr + dr[k]
                nc = cc + dc[k]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                cv = np.int64(costs[nr * cols + nc])
                if cu >= _S or cv >= _S:
                    continue
                w = cu + cv

                n_br = nr // 4
                n_bc = nc // 4
                n_local = (nr % 4) * 4 + (nc % 4)

                if cache[n_br, n_bc, 0] == -np.int64(1):
                    D_n = all_D[n_br * blocks_c + n_bc]
                    s_bids = block_bids[n_br, n_bc]
                    is_gb = (n_br == g_br) and (n_bc == g_bc)

                    # Fetch 12 × n_vg mid-distances from big table ONCE
                    for a in range(12):
                        ba = s_bids[a]
                        for vb in range(n_vg):
                            mid[a, vb] = np.int64(boundary_dist[ba, vg_bid[vb]])

                    for cell in range(16):
                        best = _S
                        for a in range(12):
                            d_ca = np.int64(D_n[cell, BI[a]])
                            if d_ca >= _S:
                                continue
                            for vb in range(n_vg):
                                m = mid[a, vb]
                                if m >= _S:
                                    continue
                                val = d_ca + m + vg_d[vb]
                                if val < best:
                                    best = val
                        if is_gb:
                            ld = np.int64(D_n[cell, g_local])
                            if ld < best:
                                best = ld
                        cache[n_br, n_bc, cell] = best

                d_ng = cache[n_br, n_bc, n_local]
                if d_ng >= _S:
                    continue
                total = w + d_ng
                if total < best_cost:
                    best_cost = total
                    best_nr = nr
                    best_nc = nc

            if best_nr == -1:
                break
            path_r[length] = best_nr
            path_c[length] = best_nc
            length += 1
            cr, cc = best_nr, best_nc

        return path_r[:length], path_c[:length]

    @_njit(cache=True, parallel=True)
    def _reconstruct_parallel(costs, rows, cols, connectivity,  # type: ignore[misc]
                              sr, sc, gr, gc,
                              all_D, boundary_dist, block_bids, BI,
                              g_br, g_bc, g_local):
        """Parallel path reconstruction via successor-field architecture.

        Pass 1 — d(cell, g) for ALL cells (prange across blocks)
        Pass 2 — successor[cell] = argmin neighbor (prange across cells)
        Pass 3 — follow successor pointers (zero-cost per step)
        """
        _S = np.int64(536_870_912)
        blocks_r = rows // 4
        blocks_c = cols // 4
        n_blocks = blocks_r * blocks_c
        total_cells = rows * cols

        D_g = all_D[g_br * blocks_c + g_bc]
        g_bids = block_bids[g_br, g_bc]

        n_vg = np.int32(0)
        vg_d = np.empty(12, dtype=np.int64)
        vg_bid = np.empty(12, dtype=np.int32)
        for b in range(12):
            d = np.int64(D_g[BI[b], g_local])
            if d < _S:
                vg_d[n_vg] = d
                vg_bid[n_vg] = g_bids[b]
                n_vg += 1

        # --- Pass 1: d(cell, g) for every cell, parallel across blocks ---
        dist_to_goal = np.full(total_cells, _S, dtype=np.int64)

        for block_idx in _prange(n_blocks):
            br = block_idx // blocks_c
            bc = block_idx % blocks_c
            D_n = all_D[block_idx]
            s_bids = block_bids[br, bc]
            is_gb = (br == g_br) and (bc == g_bc)

            mid = np.empty((12, 12), dtype=np.int64)
            for a in range(12):
                ba = s_bids[a]
                for vb in range(n_vg):
                    mid[a, vb] = np.int64(boundary_dist[ba, vg_bid[vb]])

            for cell in range(16):
                best = _S
                for a in range(12):
                    d_ca = np.int64(D_n[cell, BI[a]])
                    if d_ca >= _S:
                        continue
                    for vb in range(n_vg):
                        m = mid[a, vb]
                        if m >= _S:
                            continue
                        val = d_ca + m + vg_d[vb]
                        if val < best:
                            best = val
                if is_gb:
                    ld = np.int64(D_n[cell, g_local])
                    if ld < best:
                        best = ld

                cr = br * 4 + cell // 4
                cc = bc * 4 + cell % 4
                dist_to_goal[cr * cols + cc] = best

        # --- Pass 2: build successor field, parallel across cells ---
        successor = np.full(total_cells, np.int32(-1), dtype=np.int32)

        if connectivity == 8:
            dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:
            dr = np.array([-1, 0, 0, 1], dtype=np.int32)
            dc = np.array([0, -1, 1, 0], dtype=np.int32)
        n_nbrs = len(dr)

        for idx in _prange(total_cells):
            r = idx // cols
            c = idx % cols
            cu = np.int64(costs[idx])
            if cu >= _S:
                continue

            best_next = np.int32(-1)
            best_score = _S

            for k in range(n_nbrs):
                nr = r + dr[k]
                nc = c + dc[k]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                nidx = nr * cols + nc
                cn = np.int64(costs[nidx])
                if cn >= _S:
                    continue
                score = cu + cn + dist_to_goal[nidx]
                if score < best_score:
                    best_score = score
                    best_next = np.int32(nidx)

            successor[idx] = best_next

        # --- Pass 3: follow pointers ---
        max_len = total_cells
        path_r = np.empty(max_len, dtype=np.int32)
        path_c = np.empty(max_len, dtype=np.int32)
        path_r[0] = np.int32(sr)
        path_c[0] = np.int32(sc)
        length = np.int32(1)

        cur = sr * cols + sc
        goal_idx = gr * cols + gc

        while cur != goal_idx and length < max_len:
            nxt = successor[cur]
            if nxt < 0:
                break
            path_r[length] = np.int32(nxt // cols)
            path_c[length] = np.int32(nxt % cols)
            length += np.int32(1)
            cur = nxt

        return path_r[:length], path_c[:length]

    @_njit(cache=True)
    def _jit_reconstruct(costs, rows, cols, connectivity,  # type: ignore[misc]
                         sr, sc, gr, gc, block_dists):
        _S = np.int64(536_870_912)
        max_len = rows * cols
        path_r = np.empty(max_len, dtype=np.int32)
        path_c = np.empty(max_len, dtype=np.int32)
        path_r[0] = sr
        path_c[0] = sc
        length = 1

        if connectivity == 8:
            dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:
            dr = np.array([-1, 0, 0, 1], dtype=np.int32)
            dc = np.array([0, -1, 1, 0], dtype=np.int32)

        n_nbrs = len(dr)
        cr, cc = sr, sc

        while cr != gr or cc != gc:
            best_nr = np.int32(-1)
            best_nc = np.int32(-1)
            best_cost = _S
            cu = np.int64(costs[cr * cols + cc])
            for k in range(n_nbrs):
                nr = cr + dr[k]
                nc = cc + dc[k]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                cv = np.int64(costs[nr * cols + nc])
                if cu >= _S or cv >= _S:
                    continue
                w = cu + cv
                d_ng = block_dists[nr // 4, nc // 4, (nr % 4) * 4 + (nc % 4)]
                if d_ng >= _S:
                    continue
                total = w + d_ng
                if total < best_cost:
                    best_cost = total
                    best_nr = nr
                    best_nc = nc
            if best_nr == -1:
                break
            path_r[length] = best_nr
            path_c[length] = best_nc
            length += 1
            cr, cc = best_nr, best_nc

        return path_r[:length], path_c[:length]


# ===================================================================
# LINKED-LIST PATH RETRIEVAL (JIT)
# ===================================================================

if _HAS_NUMBA:
    @_njit(cache=True)
    def _walk_within_block(s_local, g_local, br, bc, D_block,  # type: ignore[misc]
                           costs_flat, N, NBRS, path_r, path_c, offset):
        """Greedy walk inside a 16-cell block using exact D matrix. Returns new offset."""
        _S = np.int64(536_870_912)
        blocks_c = N // 4
        cur = s_local
        path_r[offset] = np.int32(br * 4 + cur // 4)
        path_c[offset] = np.int32(bc * 4 + cur % 4)
        off = offset + 1
        for _ in range(16):
            if cur == g_local:
                break
            best_n = np.int32(-1)
            best_cost = _S
            for k in range(8):
                n = NBRS[cur, k]
                if n < 0:
                    break
                cr = br * 4 + cur // 4
                cc = bc * 4 + cur % 4
                nr = br * 4 + n // 4
                nc = bc * 4 + n % 4
                ci = costs_flat[cr * N + cc]
                cj = costs_flat[nr * N + nc]
                if ci >= _S or cj >= _S:
                    continue
                w = ci + cj
                rem = np.int64(D_block[n, g_local])
                if rem >= _S:
                    continue
                total = w + rem
                if total < best_cost:
                    best_cost = total
                    best_n = n
            if best_n < 0:
                break
            path_r[off] = np.int32(br * 4 + best_n // 4)
            path_c[off] = np.int32(bc * 4 + best_n % 4)
            off += 1
            cur = best_n
        return off

    @_njit(cache=True)
    def _jit_retrieve_linked(costs_flat, N, connectivity,  # type: ignore[misc]
                                sr, sc, gr, gc,
                                all_D, boundary_dist, block_bids,
                                BI, NBRS, LOCAL_TO_BI,
                                atlas, boundary_successor,
                                boundary_coords, boundary_local_bi):
        """Linked-list path retrieval: gateway + successor pointers + atlas."""
        _S = np.int64(536_870_912)
        blocks_c = N // 4
        max_len = N * N
        path_r = np.empty(max_len, dtype=np.int32)
        path_c = np.empty(max_len, dtype=np.int32)

        s_br = sr // 4
        s_bc = sc // 4
        g_br = gr // 4
        g_bc = gc // 4
        s_local = (sr % 4) * 4 + (sc % 4)
        g_local = (gr % 4) * 4 + (gc % 4)

        D_s = all_D[s_br * blocks_c + s_bc]
        D_g = all_D[g_br * blocks_c + g_bc]

        if s_br == g_br and s_bc == g_bc:
            off = _walk_within_block(s_local, g_local, s_br, s_bc,
                                     D_s, costs_flat, N, NBRS,
                                     path_r, path_c, 0)
            return path_r[:off], path_c[:off]

        # Gateway selection
        best_dist = _S
        best_a = np.int32(-1)
        best_b = np.int32(-1)
        best_a_bid = np.int32(-1)
        best_b_bid = np.int32(-1)
        s_base_bid = (s_br * blocks_c + s_bc) * 12
        g_base_bid = (g_br * blocks_c + g_bc) * 12

        for a in range(12):
            d_sa = np.int64(D_s[s_local, BI[a]])
            if d_sa >= _S:
                continue
            a_bid = s_base_bid + a
            for b in range(12):
                d_bg = np.int64(D_g[BI[b], g_local])
                if d_bg >= _S:
                    continue
                b_bid = g_base_bid + b
                d_mid = np.int64(boundary_dist[a_bid, b_bid])
                if d_mid >= _S:
                    continue
                total = d_sa + d_mid + d_bg
                if total < best_dist:
                    best_dist = total
                    best_a = a
                    best_b = b
                    best_a_bid = a_bid
                    best_b_bid = b_bid

        if best_dist >= _S:
            return path_r[:0], path_c[:0]

        a_cell_local = BI[best_a]
        b_cell_local = BI[best_b]
        off = np.int32(0)

        # First mile: s → a*
        if s_local != a_cell_local:
            off = _walk_within_block(s_local, a_cell_local, s_br, s_bc,
                                     D_s, costs_flat, N, NBRS,
                                     path_r, path_c, 0)
        else:
            path_r[0] = np.int32(sr)
            path_c[0] = np.int32(sc)
            off = np.int32(1)

        # Highway: a* → b* via successor pointers
        # off currently points past a* (last cell of first mile)
        # We need to avoid double-appending a*. The first mile already ends on a*.
        # Start the highway from a*, but skip re-appending a* since it's already there.
        current = best_a_bid

        for _safety in range(max_len):
            if current == best_b_bid:
                break

            nxt = boundary_successor[best_b_bid, current]
            if nxt < 0:
                break

            cr = boundary_coords[current, 0]
            cc = boundary_coords[current, 1]
            cur_br = cr // 4
            cur_bc = cc // 4
            nxt_r = boundary_coords[nxt, 0]
            nxt_c = boundary_coords[nxt, 1]
            nxt_br = nxt_r // 4
            nxt_bc = nxt_c // 4

            if cur_br == nxt_br and cur_bc == nxt_bc:
                cur_bi = boundary_local_bi[current]
                nxt_bi = boundary_local_bi[nxt]
                seg = atlas[cur_br, cur_bc, cur_bi, nxt_bi]
                seg_len = seg[0]
                # Append cells after current (index 1..seg_len-1 → local cells seg[2..seg_len])
                for i in range(1, seg_len):
                    local_cell = seg[i + 1]
                    path_r[off] = np.int32(cur_br * 4 + local_cell // 4)
                    path_c[off] = np.int32(cur_bc * 4 + local_cell % 4)
                    off += 1
            else:
                # Cross-block: just append the next boundary cell
                path_r[off] = nxt_r
                path_c[off] = nxt_c
                off += 1

            current = nxt

        # Last mile: b* → g
        if b_cell_local != g_local:
            # Use temp arrays for last mile
            tmp_r = np.empty(16, dtype=np.int32)
            tmp_c = np.empty(16, dtype=np.int32)
            lm_len = _walk_within_block(b_cell_local, g_local, g_br, g_bc,
                                        D_g, costs_flat, N, NBRS,
                                        tmp_r, tmp_c, 0)
            # Skip first cell (b*, already in path)
            for i in range(1, lm_len):
                path_r[off] = tmp_r[i]
                path_c[off] = tmp_c[i]
                off += 1

        return path_r[:off], path_c[:off]


# ===================================================================
# PUBLIC API
# ===================================================================

def precompute_all_block_dists(
    oracle: Oracle,
    goal: tuple[int, int],
) -> np.ndarray:
    """Precompute d(cell, goal) for every cell in every base block.

    Pure table lookups — zero hierarchy traversal at query time.
    Each cell's distance is:
      min_{a,b}( D[cell, BI[a]] + boundary_dist[bid_a, bid_b] + D_g[BI[b], g_local] )

    Returns int64 array of shape (blocks_r, blocks_c, 16).
    """
    gr, gc = goal
    g_br, g_bc = gr // 4, gc // 4
    g_local = (gr % 4) * 4 + (gc % 4)
    bb_g = oracle.base_blocks[g_br][g_bc]
    D_g = bb_g.D
    g_bids = oracle.block_bids[g_br, g_bc]

    blocks_r = len(oracle.base_blocks)
    blocks_c = len(oracle.base_blocks[0])

    if _HAS_NUMBA:
        return _block_dists_flat(
            oracle.all_D, oracle.boundary_dist, oracle.block_bids,
            blocks_r, blocks_c, _BI,
            g_br, g_bc, g_local, g_bids, D_g,
        )

    result = np.full((blocks_r, blocks_c, 16), SENTINEL, dtype=np.int64)
    bd = oracle.boundary_dist
    bids = oracle.block_bids
    for idx in range(blocks_r * blocks_c):
        br, bc = divmod(idx, blocks_c)
        D = oracle.all_D[idx]
        s_bids = bids[br, bc]
        for cell in range(16):
            best = SENTINEL
            for a in range(12):
                d_ca = int(D[cell, BOUNDARY_INDICES[a]])
                if d_ca >= SENTINEL:
                    continue
                ba = int(s_bids[a])
                for b in range(12):
                    d_bg = int(D_g[BOUNDARY_INDICES[b], g_local])
                    if d_bg >= SENTINEL:
                        continue
                    d_mid = int(bd[ba, int(g_bids[b])])
                    if d_mid >= SENTINEL:
                        continue
                    total = d_ca + d_mid + d_bg
                    if total < best:
                        best = total
            result[br, bc, cell] = best
        if br == g_br and bc == g_bc:
            for cell in range(16):
                ld = int(D[cell, g_local])
                if ld < result[br, bc, cell]:
                    result[br, bc, cell] = ld
    return result


def query_distance(
    oracle: Oracle,
    s: tuple[int, int],
    g: tuple[int, int],
    block_dists: np.ndarray | None = None,
) -> int:
    if s == g:
        return 0
    if block_dists is not None:
        sr, sc = s
        return int(block_dists[sr // 4, sc // 4, (sr % 4) * 4 + (sc % 4)])

    sr, sc = s
    gr, gc = g
    s_br, s_bc = sr // 4, sc // 4
    g_br, g_bc = gr // 4, gc // 4
    s_local = (sr % 4) * 4 + (sc % 4)
    g_local = (gr % 4) * 4 + (gc % 4)

    D_s = oracle.base_blocks[s_br][s_bc].D
    D_g = oracle.base_blocks[g_br][g_bc].D

    if _HAS_NUMBA:
        best = int(_query_flat(
            D_s, D_g, s_local, g_local,
            oracle.block_bids[s_br, s_bc], oracle.block_bids[g_br, g_bc],
            oracle.boundary_dist, _BI,
        ))
    else:
        best = SENTINEL
        s_bids = oracle.block_bids[s_br, s_bc]
        g_bids = oracle.block_bids[g_br, g_bc]
        bd = oracle.boundary_dist
        for a in range(12):
            d_sa = int(D_s[s_local, BOUNDARY_INDICES[a]])
            if d_sa >= SENTINEL:
                continue
            ba = int(s_bids[a])
            for b in range(12):
                d_bg = int(D_g[BOUNDARY_INDICES[b], g_local])
                if d_bg >= SENTINEL:
                    continue
                d_mid = int(bd[ba, int(g_bids[b])])
                if d_mid >= SENTINEL:
                    continue
                total = d_sa + d_mid + d_bg
                if total < best:
                    best = total

    if s_br == g_br and s_bc == g_bc:
        local_d = int(D_s[s_local, g_local])
        if local_d < best:
            best = local_d

    return min(best, SENTINEL)


def reconstruct_path(
    oracle: Oracle,
    s: tuple[int, int],
    g: tuple[int, int],
    block_dists: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if s == g:
        return [s]

    d_sg = query_distance(oracle, s, g, block_dists=block_dists)
    if d_sg >= SENTINEL:
        return []

    grid = oracle.grid
    gr, gc = g

    # Use linked-list retrieval when structures are available
    if _HAS_NUMBA and oracle.boundary_successor is not None:
        NBRS = _NBRS_8 if grid.connectivity == 8 else _NBRS_4
        pr, pc = _jit_retrieve_linked(
            oracle.costs_flat, oracle.N, grid.connectivity,
            s[0], s[1], gr, gc,
            oracle.all_D, oracle.boundary_dist, oracle.block_bids,
            _BI, NBRS, _LOCAL_TO_BI,
            oracle.atlas, oracle.boundary_successor,
            oracle.boundary_coords, oracle.boundary_local_bi,
        )
        return list(zip(pr.tolist(), pc.tolist()))

    # Fallback: inline greedy walk
    g_br, g_bc = gr // 4, gc // 4
    g_local = (gr % 4) * 4 + (gc % 4)

    if _HAS_NUMBA:
        if block_dists is not None:
            pr, pc = _jit_reconstruct(
                oracle.costs_flat, grid.rows, grid.cols, grid.connectivity,
                s[0], s[1], gr, gc, block_dists,
            )
        else:
            pr, pc = _jit_reconstruct_inline(
                oracle.costs_flat, grid.rows, grid.cols, grid.connectivity,
                s[0], s[1], gr, gc,
                oracle.all_D, oracle.boundary_dist, oracle.block_bids, _BI,
                g_br, g_bc, g_local,
            )
        return list(zip(pr.tolist(), pc.tolist()))

    if block_dists is None:
        block_dists = precompute_all_block_dists(oracle, g)

    path = [s]
    current = s
    while current != g:
        best_next = None
        best_cost = SENTINEL
        for nr, nc in grid.get_neighbors(current[0], current[1]):
            w = grid.edge_weight(current[0], current[1], nr, nc)
            if w >= SENTINEL:
                continue
            d_ng = int(block_dists[nr // 4, nc // 4, (nr % 4) * 4 + (nc % 4)])
            total = tropical_add(w, d_ng)
            if total < best_cost:
                best_cost = total
                best_next = (nr, nc)
        if best_next is None:
            break
        path.append(best_next)
        current = best_next
    return path
