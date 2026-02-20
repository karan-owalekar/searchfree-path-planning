from __future__ import annotations

from pathcircuit.grid import Grid
from pathcircuit.core.base_block import BaseBlock, compute_base_block
from pathcircuit.core.hierarchy import build_hierarchy
from pathcircuit.core.uncondense import un_condense_all
from pathcircuit.core.oracle import Oracle, build_oracle


def build_pathcircuit_oracle(grid: Grid) -> Oracle:
    """Full pipeline: pad grid -> base blocks -> hierarchy -> un-condense -> oracle."""
    padded = grid.to_padded()
    N = padded.rows  # square after padding

    blocks_r = N // 4
    blocks_c = N // 4

    base_blocks: list[list[BaseBlock]] = []
    for br in range(blocks_r):
        row: list[BaseBlock] = []
        for bc in range(blocks_c):
            row.append(compute_base_block(br, bc, padded))
        base_blocks.append(row)

    if N == 4:
        bb = base_blocks[0][0]
        bb.D_global = bb.T.copy()
        bb.D_global_cells = list(bb.boundary_cells)
        bb.D_global_cell_to_index = dict(bb.boundary_cell_to_index)
        return _build_single_block_oracle(bb, base_blocks, N, padded)

    root, all_nodes = build_hierarchy(base_blocks, N, padded)
    un_condense_all(root, N)

    return build_oracle(root, base_blocks, N, padded)


def _build_single_block_oracle(
    bb: BaseBlock,
    base_blocks: list[list[BaseBlock]],
    N: int,
    grid: Grid,
) -> Oracle:
    """Trivial oracle for a single 4x4 base block (no hierarchy)."""
    return Oracle(
        base_blocks=base_blocks,
        N=N,
        grid=grid,
    )
