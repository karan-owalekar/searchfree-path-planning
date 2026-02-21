from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pathcircuit.grid import Grid
from pathcircuit.core.tropical import SENTINEL
from pathcircuit.core.base_block import BaseBlock, compute_base_block
from pathcircuit.core.hierarchy import build_hierarchy
from pathcircuit.core.uncondense import un_condense_all
from pathcircuit.core.oracle import Oracle, build_oracle


def build_pathcircuit_oracle(
    grid: Grid,
    *,
    build_linked_list: bool = True,
    use_gpu: bool = False,
) -> Oracle:
    """Full pipeline: pad grid -> base blocks -> hierarchy -> un-condense -> oracle.

    Args:
        build_linked_list: If True, build atlas + successor table for ~2μs path
            retrieval. If False, skip them for faster oracle build (path
            reconstruction falls back to inline greedy walk).
        use_gpu: If True, build boundary table on GPU via CUDA.
    """
    padded = grid.to_padded()
    N = padded.rows  # square after padding

    blocks_r = N // 4
    blocks_c = N // 4

    n_blocks = blocks_r * blocks_c
    flat_blocks: list[BaseBlock | None] = [None] * n_blocks

    import numpy as _np
    costs_flat = padded.costs.ravel().astype(_np.int32)

    if n_blocks >= 64:
        n_workers = min(os.cpu_count() or 4, n_blocks)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for br in range(blocks_r):
                for bc in range(blocks_c):
                    fut = pool.submit(compute_base_block, br, bc, padded,
                                      _costs_flat=costs_flat)
                    futures[fut] = (br, bc)
            for fut in futures:
                br, bc = futures[fut]
                flat_blocks[br * blocks_c + bc] = fut.result()
    else:
        for br in range(blocks_r):
            for bc in range(blocks_c):
                flat_blocks[br * blocks_c + bc] = compute_base_block(
                    br, bc, padded, _costs_flat=costs_flat)

    base_blocks: list[list[BaseBlock]] = []
    for br in range(blocks_r):
        row: list[BaseBlock] = []
        for bc in range(blocks_c):
            row.append(flat_blocks[br * blocks_c + bc])  # type: ignore[arg-type]
        base_blocks.append(row)

    if N == 4:
        bb = base_blocks[0][0]
        bb.D_global = bb.T.copy()
        bb.D_global_cells = list(bb.boundary_cells)
        bb.D_global_cell_to_index = dict(bb.boundary_cell_to_index)
        return _build_single_block_oracle(bb, base_blocks, N, padded,
                                          build_linked_list=build_linked_list)

    root, all_nodes = build_hierarchy(base_blocks, N, padded)
    un_condense_all(root, N)

    return build_oracle(root, base_blocks, N, padded,
                        build_linked_list=build_linked_list,
                        use_gpu=use_gpu)


def _build_single_block_oracle(
    bb: BaseBlock,
    base_blocks: list[list[BaseBlock]],
    N: int,
    grid: Grid,
    *,
    build_linked_list: bool = True,
) -> Oracle:
    """Trivial oracle for a single 4x4 base block (no hierarchy)."""
    all_D = bb.D[np.newaxis, :, :].astype(np.int32)

    boundary_dist = np.full((12, 12), SENTINEL, dtype=np.int32)
    if bb.D_global is not None:
        np.minimum(boundary_dist, bb.D_global[:12, :12].astype(np.int32),
                   out=boundary_dist)

    block_bids = np.arange(12, dtype=np.int32).reshape(1, 1, 12)
    costs_flat = grid.costs.ravel().astype(np.int64)

    atlas = successor = boundary_coords = boundary_local_bi = None
    if build_linked_list:
        from pathcircuit.core.oracle import _build_linked_list_structures
        atlas, successor, boundary_coords, boundary_local_bi = \
            _build_linked_list_structures(base_blocks, 1, 1, N, grid,
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
