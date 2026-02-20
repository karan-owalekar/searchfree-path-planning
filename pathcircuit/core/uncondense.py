from __future__ import annotations

import math

import numpy as np

from pathcircuit.core.tropical import tropical_matmul
from pathcircuit.core.base_block import BaseBlock
from pathcircuit.core.hierarchy import MergeNode, BlockLike


def un_condense_single_merge(node: MergeNode) -> None:
    """Recover global distances for the interface vertices eliminated in this merge."""
    nE = node.boundary_size
    nS = node.interface_size

    if nS == 0:
        return

    D_EE = node.D_global  # nE x nE, set by parent propagation
    assert D_EE is not None

    M_ES_MSS = node.M_ES_times_MSS_star  # nE x nS
    M_ES = node.M_ES                      # nE x nS
    M_SE = node.M_SE                      # nS x nE
    M_SS_star = node.M_SS_star            # nS x nS

    # D_ES = D_EE ⊗ M_ES_MSS
    D_ES = tropical_matmul(D_EE, M_ES_MSS)

    # D_SE = D_ES^T (symmetric distances)
    D_SE = D_ES.T.copy()

    # D_SS = M_SS_star ⊕ (M_SE ⊗ D_EE ⊗ M_ES)
    temp = tropical_matmul(M_SE, D_EE)
    temp2 = tropical_matmul(temp, M_ES)
    D_SS = np.minimum(M_SS_star, temp2)

    # Assemble full distance matrix for E ∪ S
    total = nE + nS
    D_full = np.full((total, total), 0, dtype=np.int32)
    D_full[:nE, :nE] = D_EE
    D_full[:nE, nE:total] = D_ES
    D_full[nE:total, :nE] = D_SE
    D_full[nE:total, nE:total] = D_SS

    node.D_global = D_full
    node.D_global_cells = list(node.boundary_cells) + list(node.interface_cells)
    node.D_global_cell_to_index = {c: i for i, c in enumerate(node.D_global_cells)}


def propagate_D_global(parent: MergeNode, child: BlockLike) -> None:
    """Extract the sub-matrix of parent.D_global corresponding to child's boundary cells."""
    assert parent.D_global is not None
    assert parent.D_global_cell_to_index is not None

    indices: list[int] = []
    for cell in child.boundary_cells:
        if cell not in parent.D_global_cell_to_index:
            raise RuntimeError(
                f"Child boundary cell {cell} not found in parent D_global "
                f"(parent region: top={parent.region_top}, left={parent.region_left})"
            )
        indices.append(parent.D_global_cell_to_index[cell])

    n = len(indices)
    idx = np.array(indices)
    child_D = parent.D_global[np.ix_(idx, idx)].copy()

    child.D_global = child_D
    child.D_global_cells = list(child.boundary_cells)
    child.D_global_cell_to_index = {c: i for i, c in enumerate(child.boundary_cells)}


def un_condense_all(root: MergeNode, N: int) -> None:
    """Top-down un-condensation over the full hierarchy."""
    L = int(math.log2(N / 4))

    # Initialize root: its D_global is just its own transfer matrix
    root.D_global = root.T.copy()
    root.D_global_cells = list(root.boundary_cells)
    root.D_global_cell_to_index = dict(root.boundary_cell_to_index)

    _un_condense_subtree(root)


def _un_condense_subtree(node: MergeNode) -> None:
    """Recursively un-condense a subtree rooted at *node*."""
    # Step A: un-condense this node's own merge (recover interface distances)
    un_condense_single_merge(node)

    if node.h_merge_left is not None and node.h_merge_right is not None:
        # This is a vertical merge node with horizontal sub-merges AB and CD
        AB = node.h_merge_left
        CD = node.h_merge_right

        # Propagate D_global to AB and CD
        propagate_D_global(node, AB)
        propagate_D_global(node, CD)

        # Un-condense the horizontal merges
        un_condense_single_merge(AB)
        un_condense_single_merge(CD)

        # Propagate to their children and recurse
        _propagate_and_recurse(AB)
        _propagate_and_recurse(CD)
    else:
        # Direct children of a merge without sub-merges (shouldn't happen at
        # level > 0 in our tree, but handle gracefully)
        _propagate_and_recurse(node)


def _propagate_and_recurse(node: MergeNode) -> None:
    """Propagate D_global from *node* to its left/right children, recurse if MergeNode."""
    for child in (node.child_left, node.child_right):
        propagate_D_global(node, child)
        if isinstance(child, MergeNode):
            _un_condense_subtree(child)
