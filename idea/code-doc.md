# Implementation Documentation: Search-Free Path Planner

This document details how the search-free algebraic pathfinding algorithm is implemented in code. It covers every module in `pathcircuit/core/` and `pathcircuit/grid.py`, explaining what each piece does, why it was done that way, and how the parts connect.

---

## Table of Contents

1. [Project Layout](#1-project-layout)
2. [Dependencies and Why They Were Chosen](#2-dependencies-and-why-they-were-chosen)
3. [Grid Representation](#3-grid-representation)
4. [Tropical Semiring Operations](#4-tropical-semiring-operations)
5. [Base Block Computation](#5-base-block-computation)
6. [Hierarchical Composition](#6-hierarchical-composition)
7. [Un-Condensation](#7-un-condensation)
8. [Oracle and Query System](#8-oracle-and-query-system)
9. [Pipeline Orchestration](#9-pipeline-orchestration)
10. [Design Decisions and Trade-offs](#10-design-decisions-and-trade-offs)

---

## 1. Project Layout

The algorithm lives entirely within the `pathcircuit/` package. The core computation is separated from the UI and comparison modules:

```
pathcircuit/
├── grid.py              # Grid data structure (shared by all modules)
├── core/
│   ├── tropical.py      # Tropical semiring primitives (add, min, matmul)
│   ├── base_block.py    # Phase 1: 4×4 base block APSP + transfer matrices
│   ├── hierarchy.py     # Phase 2: hierarchical merge (Schur complement)
│   ├── uncondense.py    # Phase 3a: top-down distance recovery
│   ├── oracle.py        # Phase 3b: ancestor chains, expansion, queries
│   └── pipeline.py      # Orchestrator: ties all phases together
```

The algorithm proceeds in four sequential phases:

1. **Base block computation** — solve all-pairs shortest paths within every 4×4 tile.
2. **Hierarchical composition** — merge tiles bottom-up using the tropical Schur complement.
3. **Un-condensation** — propagate global distances top-down through the hierarchy.
4. **Oracle construction** — build ancestor chains for search-free queries.

---

## 2. Dependencies and Why They Were Chosen

Defined in `pyproject.toml`:

| Dependency | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.24 | All matrix operations (tropical matmul, Kleene closure, distance tables). NumPy's vectorised broadcasting turns the O(n³) tropical matrix multiply into a single `a[:, :, np.newaxis] + b[np.newaxis, :, :]` expression, which is orders of magnitude faster than triple-nested Python loops. |
| `pygame-ce` | >= 2.5 | UI only — not used by the algorithm. |

**Why NumPy instead of pure Python:** The algorithm's inner loop is tropical matrix multiplication, which is structurally identical to standard matmul but with `(min, +)` replacing `(+, ×)`. A 16×16 matmul is 4096 additions and 256 minimisations — tolerable in Python — but the hierarchy produces matrices up to ~120×120 at the root level of a 64×64 grid. At that size, pure Python would be 1000× slower than NumPy's vectorised path. Every matrix in the system (weight matrices, transfer matrices, D_global tables) is stored as `np.ndarray` with dtype `int32`.

**Why int32:** The SENTINEL value is 2²⁹ = 536,870,912. Two SENTINEL values summed stay below the int32 max of 2³¹ − 1, so overflow is impossible within int32 arithmetic. Intermediate sums during matmul are computed in int64 to avoid overflow when summing two non-SENTINEL values that could theoretically exceed int32 range, then clipped back to SENTINEL.

---

## 3. Grid Representation

**File:** `pathcircuit/grid.py`

The `Grid` class stores the N×M grid as a 2D NumPy array of integer cell costs:

```python
self.costs = np.ones((rows, cols), dtype=np.int32)
```

Each cell holds a cost value. An obstacle is represented by setting the cost to `SENTINEL` (2²⁹). This avoids needing a separate boolean obstacle array — a cell is an obstacle if and only if `costs[r, c] >= SENTINEL`.

### Edge Weights

The edge weight between two adjacent cells is the sum of their costs:

```python
def edge_weight(self, r1, c1, r2, c2) -> int:
    ci = int(self.costs[r1, c1])
    cj = int(self.costs[r2, c2])
    if ci >= SENTINEL or cj >= SENTINEL:
        return SENTINEL
    return ci + cj
```

This is a vertex-cost model: traversing an edge "pays" for entering both endpoints. For a uniform grid where all costs are 1, every edge has weight 2, so shortest-path distances equal twice the Manhattan/Chebyshev distance.

### Connectivity

The grid supports both 4-connected (cardinal directions only) and 8-connected (cardinal + diagonal) movement. This is set at construction time via the `connectivity` parameter (4 or 8). Neighbor enumeration uses precomputed offset lists:

```python
CARDINAL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
```

### Grid Padding

The hierarchical algorithm requires the grid dimensions to be a power-of-2 multiple of 4 (so the grid can be recursively halved into 4×4 base blocks). The `to_padded()` method computes the smallest valid dimension ≥ max(rows, cols) and creates a square padded grid. Padding cells are filled with SENTINEL (treated as obstacles), so they never participate in shortest paths.

The valid dimension formula: find the smallest `n` such that `n >= d`, `n` is a multiple of 4, and `n / 4` is a power of 2. For example, a 12×12 grid pads to 16×16 (16/4 = 4 = 2²). A 20×20 grid pads to 32×32 (32/4 = 8 = 2³).

---

## 4. Tropical Semiring Operations

**File:** `pathcircuit/core/tropical.py`

The entire algorithm is built on the **tropical semiring** (ℝ≥0 ∪ {∞}, min, +), where:
- "Addition" is `min(a, b)`
- "Multiplication" is `a + b`
- The additive identity (zero) is `∞` (SENTINEL)
- The multiplicative identity (one) is `0`

### Scalar Operations

```python
SENTINEL = 536_870_912  # 2^29, represents ∞

def tropical_add(a, b):
    # a ⊕ b in the tropical semiring = ordinary a + b
    # Returns SENTINEL if either input is SENTINEL (∞ + x = ∞)
    # Clamps overflow to SENTINEL

def tropical_min(a, b):
    # min(a, b) — the "addition" in the tropical semiring
    return a if a <= b else b
```

The early-exit on SENTINEL in `tropical_add` prevents computing `SENTINEL + x` which could overflow even in int64.

### Matrix Multiplication

```python
def tropical_matmul(A, B):
    # C[i,j] = min_k (A[i,k] + B[k,j])
```

This is the core computational primitive. Instead of a triple-nested loop, it uses NumPy broadcasting:

```python
a = A.astype(np.int64)
b = B.astype(np.int64)
sums = a[:, :, np.newaxis] + b[np.newaxis, :, :]  # shape (n, m, p)
sums = np.clip(sums, 0, SENTINEL)
C = np.min(sums, axis=1).astype(np.int32)
```

This creates a 3D tensor of all pairwise sums and then takes the minimum along the middle axis. The int64 upcast prevents overflow during the addition step; clipping to `[0, SENTINEL]` ensures the result stays in valid range before downcasting to int32.

**Memory trade-off:** For an n×n matrix, this allocates an O(n³) intermediate tensor. For the sizes in this algorithm (n ≤ ~120), this is at most ~14 MB — acceptable on modern hardware and far faster than interpreted loops.

### Witness Recording

```python
def tropical_matmul_with_witness(A, B):
    # Returns (C, W) where C[i,j] = min_k(A[i,k]+B[k,j]) and W[i,j] = argmin_k
```

The witness matrix `W` records *which* intermediate index `k` achieved the minimum. This is used for path reconstruction via recursive bisection: if the shortest path from `i` to `j` goes through midpoint `k = W[i,j]`, it can be decomposed into subpaths `i→k` and `k→j`. The witnesses are stored at each level for potential future use, though the current implementation uses greedy reconstruction instead.

---

## 5. Base Block Computation

**File:** `pathcircuit/core/base_block.py`

This is Phase 1 of the algorithm. The padded N×N grid is divided into (N/4)² non-overlapping 4×4 tiles called **base blocks**. For each base block, we compute:
1. The 16×16 all-pairs shortest path matrix `D`
2. The 12×12 **transfer matrix** `T` (a condensed summary of boundary-to-boundary distances)

### Cell Indexing

The 16 cells in a 4×4 block are numbered in row-major order (0–15):

```
 0  1  2  3       B  B  B  B
 4  5  6  7       B  ι  ι  B
 8  9 10 11       B  ι  ι  B
12 13 14 15       B  B  B  B
```

Of these 16 cells, 12 are **boundary cells** (on the perimeter of the block) and 4 are **interior cells** (the 2×2 centre):

```python
BOUNDARY_INDICES = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
INTERIOR_INDICES = [5, 6, 9, 10]
```

### Intra-Block Adjacency

The adjacency within a 4×4 block is precomputed as a dictionary mapping each local cell index to its list of neighbours. For 4-connectivity, `NEIGHBORS_4` lists cardinal neighbours. For 8-connectivity, `NEIGHBORS_8` extends this with diagonal neighbours. These are computed once at module load time:

```python
NEIGHBORS_8: dict[int, list[int]] = {}
for _cell in range(16):
    _r, _c = divmod(_cell, 4)
    _nbrs = list(NEIGHBORS_4[_cell])
    for _dr, _dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < 4 and 0 <= _nc < 4:
            _nbrs.append(_nr * 4 + _nc)
    NEIGHBORS_8[_cell] = _nbrs
```

### Weight Matrix Construction

`build_weight_matrix(br, bc, grid)` creates the 16×16 adjacency matrix `W` for block `(br, bc)`:
- `W[i, i] = 0` (self-loops cost nothing)
- `W[i, j] = cost(i) + cost(j)` if `i` and `j` are adjacent, respecting connectivity mode
- `W[i, j] = SENTINEL` otherwise

### All-Pairs Shortest Paths via Repeated Squaring

Instead of running Dijkstra or Floyd-Warshall 16 times, the implementation computes APSP by **repeated squaring** of the weight matrix under tropical multiplication:

```python
S1, w1 = tropical_matmul_with_witness(W, W)      # W² — paths of length ≤ 2
S2, w2 = tropical_matmul_with_witness(S1, S1)    # W⁴ — paths of length ≤ 4
S3, w3 = tropical_matmul_with_witness(S2, S2)    # W⁸ — paths of length ≤ 8
D,  w4 = tropical_matmul_with_witness(S3, S3)    # W¹⁶ — paths of length ≤ 16
```

After 4 squarings, `D = W¹⁶`. Since the longest simple path in a 4×4 grid has at most 15 edges (visiting all 16 cells), W¹⁶ captures all shortest paths. The key insight: if `W` has zero diagonal (which it does), then `W^k[i, j] ≤ W^(k-1)[i, j]` for all k, so the sequence is monotonically non-increasing and stabilises once the exponent exceeds the graph diameter.

**Why repeated squaring over Floyd-Warshall:** Both are O(n³ log n) vs O(n³) for a fixed n=16, so the constant-factor difference is negligible. But repeated squaring produces witness matrices for free that enable recursive path bisection, and it maps cleanly to the vectorised NumPy matmul.

### Transfer Matrix Extraction

The **transfer matrix** `T` is a 12×12 submatrix of `D` containing only the boundary-to-boundary distances:

```python
T[a, b] = D[BOUNDARY_INDICES[a], BOUNDARY_INDICES[b]]
```

This is the block's "summary" for the hierarchical merge: the interior cells have been implicitly eliminated, and `T` encodes how quickly you can transit between any two perimeter cells, routing through the interior as needed.

### The BaseBlock Data Structure

```python
@dataclass
class BaseBlock:
    block_row: int                              # block grid position
    block_col: int
    D: np.ndarray                               # 16×16 APSP matrix
    T: np.ndarray                               # 12×12 transfer matrix
    witnesses: list[np.ndarray]                 # 4 witness matrices from squaring
    boundary_cells: list[tuple[int, int]]       # 12 global (row, col) coords
    boundary_cell_to_index: dict[...]           # reverse lookup
    region_top: int                             # top-left corner in global grid
    region_left: int
    D_global: np.ndarray | None                 # filled during un-condensation
    D_global_cells: list[...] | None
    D_global_cell_to_index: dict[...] | None
```

The `boundary_cells` list maps boundary index (0–11) to global grid coordinates. `boundary_cell_to_index` provides the reverse mapping. These are essential during the merge phase, where cells from different blocks need to be matched by their global position.

---

## 6. Hierarchical Composition

**File:** `pathcircuit/core/hierarchy.py`

This is Phase 2. Base blocks are merged bottom-up in a quadtree pattern: four adjacent blocks form a 2×2 group, which is merged into a single larger block. This repeats until the entire grid is a single block at the root.

### Merge Pattern

At each level, four sub-blocks `[A, B; C, D]` are combined through three binary merges:

```
[A] [B]       Merge 1 (horizontal): A ∥ B → AB
[C] [D]       Merge 2 (horizontal): C ∥ D → CD
              Merge 3 (vertical):   AB ⫽ CD → ABCD
```

All three MergeNodes are stored because un-condensation needs them.

### The MergeNode Data Structure

```python
@dataclass
class MergeNode:
    level: int                          # hierarchy depth
    merge_type: str                     # "horizontal" or "vertical"
    region_top, region_left: int        # covered grid region
    region_height, region_width: int
    boundary_size: int                  # |E| — external boundary cell count
    boundary_cells: list[...]           # ordered boundary coordinates
    boundary_cell_to_index: dict[...]
    T: np.ndarray                       # transfer matrix for this merged block
    interface_size: int                 # |S| — interface cell count
    interface_cells: list[...]
    M_SS_star: np.ndarray               # Kleene closure of interface sub-matrix
    M_ES_times_MSS_star: np.ndarray     # precomputed product for un-condensation
    M_ES: np.ndarray                    # boundary-to-interface distances
    M_SE: np.ndarray                    # interface-to-boundary distances
    kleene_witnesses: list[np.ndarray]
    child_left: BlockLike               # left/top child
    child_right: BlockLike              # right/bottom child
    h_merge_left: MergeNode | None      # horizontal sub-merges (for vertical nodes)
    h_merge_right: MergeNode | None
    D_global: np.ndarray | None         # filled during un-condensation
    D_global_cells: list[...] | None
    D_global_cell_to_index: dict[...] | None
```

### Partition Identification

When two blocks are merged, their shared edge cells are classified:

- **Interface cells (S):** Cells on the shared edge that are *not* corners of the merged region. These will be "eliminated" (condensed out) by the Schur complement. For a horizontal merge, these are the non-corner cells on the right column of A and the left column of B.
- **External cells (E):** All boundary cells of the merged block (the outer perimeter of the combined region). These survive the merge and become the new block's boundary.

For a horizontal merge of two blocks with height h:
- |I_A| = |I_B| = h − 2 (shared-edge interior cells from each side)
- |S| = 2(h − 2) (total interface cells)
- |E| = 2h + 2(w_A + w_B) − 4 (perimeter of the combined region)

### Boundary Enumeration

The `enumerate_boundary` function produces a deterministic ordering of boundary cells for any rectangular region: top edge left-to-right, right edge top-to-bottom (excluding corners), bottom edge left-to-right, left edge top-to-bottom (excluding corners). This consistent ordering is critical — every reference to "boundary cell index i" throughout the entire hierarchy must agree on this ordering.

### Joint Matrix Assembly

The merge's central computation is the **joint matrix** `M`, an (|E| + |S|) × (|E| + |S|) matrix partitioned into four blocks:

```
M = [ M_EE  M_ES ]
    [ M_SE  M_SS ]
```

Where:
- `M_EE[i, j]`: distance between external cells `i` and `j`, looked up from the child transfer matrices if both cells belong to the same child, or set to the direct edge weight if they're grid-adjacent cells from different children.
- `M_ES[i, j]` and `M_SE[j, i]`: distance between an external cell and an interface cell, from the same child's transfer matrix. If they're from different children but grid-adjacent (relevant in 8-connectivity), the direct edge weight is used.
- `M_SS[i, j]`: distance between interface cells. Same-child pairs use the transfer matrix. Cross-child pairs use the direct edge weight if grid-adjacent (same row for horizontal merges, same column for vertical).

**8-connectivity extension:** The original spec (4-connectivity) only needed to check same-row adjacency for cross-child interface pairs. For 8-connectivity, the `are_grid_adjacent` function uses Chebyshev distance (max of absolute row and column differences equals 1) to handle diagonal connections between cells from different children. This captures diagonal cross-block edges that don't appear in either child's transfer matrix.

### Kleene Closure

The Kleene closure `M_SS*` computes all-pairs shortest paths among the interface cells:

```python
def kleene_closure(M_SS, s_size):
    num_sq = max(1, ceil(log2(s_size))) if s_size > 1 else 1
    R = M_SS.copy()
    for _ in range(num_sq):
        R_new, W_t = tropical_matmul_with_witness(R, R)
        R = R_new
    return R, witnesses
```

The number of squarings is ⌈log₂|S|⌉, which guarantees the exponent exceeds |S| (the maximum simple path length among S cells). Because the diagonal of M_SS is zero, each squaring can only decrease entries (the previous value is always ≤ the new value), so no separate accumulation step is needed — the squared result automatically includes the old result as a special case (routing through the identity element k = i).

### Tropical Schur Complement

The transfer matrix `T_C` for the merged block is computed via the tropical Schur complement:

```
T_C = M_EE ⊕ (M_ES ⊗ M_SS* ⊗ M_SE)
```

In code:

```python
M_ES_times_MSS_star = tropical_matmul(M_ES, M_SS_star)
schur_product = tropical_matmul(M_ES_times_MSS_star, M_SE)
T_C = np.minimum(M_EE, schur_product)
```

This is the algebraic analogue of Gaussian elimination: the interface cells are "eliminated" by computing the best path from any external cell to any other external cell that optionally routes through the interface. `M_EE` captures direct paths; `M_ES ⊗ M_SS* ⊗ M_SE` captures paths that detour through the interface. The element-wise minimum takes the better of the two.

`M_ES_times_MSS_star` (the product `M_ES ⊗ M_SS*`) is stored on the MergeNode because it's reused during un-condensation.

### Full Hierarchy Construction

```python
def build_hierarchy(base_blocks, N, grid):
    L = int(log2(N / 4))     # number of levels
    prev_level = base_blocks  # 2D grid of blocks

    for level in range(L):
        # Group blocks into 2×2 quads, merge each quad
        for each 2×2 group (A, B, C, D):
            AB = horizontal_merge(A, B, level, grid)
            CD = horizontal_merge(C, D, level, grid)
            ABCD = vertical_merge(AB, CD, level, grid)
            ABCD.h_merge_left = AB    # store sub-merges
            ABCD.h_merge_right = CD

    return root  # single block covering the entire grid
```

Each level halves the number of blocks in each dimension. After L levels, a single root block remains.

---

## 7. Un-Condensation

**File:** `pathcircuit/core/uncondense.py`

This is Phase 3a. During hierarchical composition, interface cells were eliminated by the Schur complement — their distances were folded into the transfer matrix but the individual cell-to-cell values were lost. Un-condensation recovers them.

### Top-Down Processing

Un-condensation works **top-down**, starting from the root:

1. The root's `D_global` is initialised to its own transfer matrix `T` (boundary-to-boundary distances).
2. At each merge node, interface distances are recovered using the stored Schur complement data.
3. The expanded `D_global` is propagated down to child nodes.

### Recovery at a Single Merge Node

Given a node with `D_global = D_EE` (boundary distances, set by its parent), the un-condensation recovers:

```
D_ES = D_EE ⊗ (M_ES ⊗ M_SS*)    — external-to-interface distances
D_SE = D_ES^T                     — by symmetry of shortest paths
D_SS = M_SS* ⊕ (M_SE ⊗ D_EE ⊗ M_ES)  — interface-to-interface distances
```

The key formula for `D_SS`: the first term `M_SS*` captures shortest paths among interface cells that stay within the local merge. The second term `M_SE ⊗ D_EE ⊗ M_ES` captures paths that leave through the boundary, traverse the global network, and re-enter. The minimum of both gives the true global shortest path.

After recovery, the node's `D_global` is expanded from |E|×|E| to (|E|+|S|)×(|E|+|S|):

```python
D_full[   :nE,    :nE  ] = D_EE
D_full[   :nE,  nE:    ] = D_ES
D_full[ nE:  ,    :nE  ] = D_SE
D_full[ nE:  ,  nE:    ] = D_SS
```

### Propagation to Children

`propagate_D_global(parent, child)` extracts the sub-matrix of `parent.D_global` corresponding to the child's boundary cells. Each child boundary cell must appear somewhere in the parent's `D_global_cells` list (as either a boundary or interface cell of the parent). The function finds these indices and extracts the corresponding rows and columns:

```python
for cell in child.boundary_cells:
    indices.append(parent.D_global_cell_to_index[cell])
child.D_global = parent.D_global[np.ix_(idx, idx)]
```

This gives the child its own `D_EE` (global boundary distances), which accounts for paths that leave the child, traverse the wider grid, and return.

### Recursive Subtree Processing

The recursion handles the three-node structure at each 2×2 group:

1. Un-condense the vertical merge node ABCD (recover its interface cells).
2. Propagate `D_global` from ABCD to its horizontal sub-merges AB and CD.
3. Un-condense AB and CD (recover their interface cells).
4. Propagate from AB to its children A and B; from CD to C and D.
5. If A, B, C, D are themselves MergeNodes (not base blocks), recurse.

---

## 8. Oracle and Query System

**File:** `pathcircuit/core/oracle.py`

This is Phase 3b. After un-condensation, every node in the hierarchy has a `D_global` table containing shortest distances among its boundary and interface cells. The oracle wraps this into a query-answerable structure.

### The Oracle Data Structure

```python
@dataclass
class Oracle:
    base_blocks: list[list[BaseBlock]]          # the (N/4)² base blocks
    N: int                                       # padded grid size
    grid: Grid                                   # the padded grid
    ancestor_chains: dict[tuple[int,int], list[MergeNode]]
```

The `ancestor_chains` dictionary is the key structure for search-free queries. For each base block `(br, bc)`, it stores the ordered list of MergeNodes from the block's immediate parent up to the root.

### Ancestor Chain Construction

`_build_ancestor_chains(root)` walks the hierarchy tree recursively. As it descends, it accumulates a `suffix` list representing the chain of ancestors from the current position up to the root. When it reaches a base block, that suffix becomes the block's ancestor chain.

The chain is built **bottom-up** by prepending nodes during recursion:

```python
def walk(node, suffix):
    if isinstance(node, BaseBlock):
        chains[(node.block_row, node.block_col)] = list(suffix)
        return
    if node.h_merge_left is not None:
        for h_merge in (node.h_merge_left, node.h_merge_right):
            child_suffix = [h_merge, node] + suffix  # prepend
            for child in (h_merge.child_left, h_merge.child_right):
                walk(child, child_suffix)
    else:
        child_suffix = [node] + suffix
        for child in (node.child_left, node.child_right):
            walk(child, child_suffix)
```

The prepend (`[h_merge, node] + suffix`) ensures the chain starts with the immediate parent and ends with the root. This ordering is critical for the expansion algorithm: the initial distances from a base block's boundary cells must first be propagated to the immediate parent's `D_global` table (where those boundary cells appear), then successively to larger and larger ancestor tables.

**Why bottom-up order matters:** An earlier implementation used top-down order (root first). This caused the expansion to start at the highest level, where the base block's boundary cells are a tiny subset of the root's large `D_global` table. The expansion had very few "bridge" cells to work with, producing wildly overestimated distances. Starting from the immediate parent guarantees all 12 boundary cells of the base block are present as bridges, giving the densest and most accurate initial expansion.

### Search-Free Hierarchical Expansion

`_expand_through_hierarchy(chain, initial)` is the core search-free operation. It projects a set of initial distances (from a source or goal cell to its base block's boundary cells) upward through every node in the ancestor chain using **min-plus matrix-vector products**.

For each node in the chain:

1. **Identify bridge cells:** cells that are both in the current distance dictionary `d` and in the node's `D_global_cells`.
2. **Vectorised expansion:** compute `d_new[k] = min_b (d[bridge_b] + D_global[bridge_b, k])` for all cells `k` in the node's D_global.
3. **Update:** if `d_new[k]` improves the known distance to cell `k`, update `d`.

In NumPy:

```python
bi = np.array(bridge_idx)
bd = np.array(bridge_d, dtype=np.int64)
D_sub = node.D_global[bi, :].astype(np.int64)
sums = bd[:, np.newaxis] + D_sub              # broadcast
np.clip(sums, 0, SENTINEL, out=sums)
d_new = np.min(sums, axis=0)
```

This is a min-plus matrix-vector product: the "matrix" is the rows of `D_global` corresponding to bridge cells, and the "vector" is the current distances to those bridge cells.

**Why this is search-free:** There are no priority queues, no visited sets, no graph traversal. It is pure algebraic computation — a sequence of matrix-vector products on precomputed tables. The number of operations is fixed by the hierarchy structure, not by the graph topology.

### Distance Query

`query_distance(oracle, s, g, goal_expansion=None)` computes the shortest path distance:

1. **Localise:** find which base block contains `s` and `g`. Convert to local indices within the 4×4 block.
2. **Same-block shortcut:** if `s` and `g` are in the same base block, read `bb.D[s_local, g_local]` as an initial upper bound.
3. **Source-side expansion:** compute distances from `s` to its base block's boundary cells using `bb_s.D[s_local, BOUNDARY_INDICES[b]]`, then expand upward through the hierarchy via `_expand_through_hierarchy`.
4. **Goal-side expansion:** same as source-side but starting from `g`'s boundary cells. This can be precomputed and reused (via the `goal_expansion` parameter) when multiple queries share the same goal.
5. **Combine:** for every cell `c` reachable from both sides, compute `d_from_s[c] + d_to_g[c]`. The minimum over all such cells is the shortest path distance.

```python
for cell, ds in d_from_s.items():
    dg = goal_expansion.get(cell, SENTINEL)
    total = ds + dg
    if total < best:
        best = total
```

### Path Reconstruction

`reconstruct_path(oracle, s, g)` uses a greedy walk:

1. Precompute the goal-side expansion once.
2. Starting at `s`, examine all neighbours of the current cell.
3. For each neighbour `n`, compute `edge_weight(current, n) + query_distance(oracle, n, g)`.
4. Step to the neighbour with the smallest total cost.
5. Repeat until reaching `g`.

This is correct because the oracle returns exact distances. By the optimality principle, the neighbour `n*` minimising `w(current, n*) + d(n*, g)` is always on a shortest path. The greedy choice never makes a suboptimal step.

**Performance note:** each step queries up to 8 neighbours, and each query does a full source-side hierarchical expansion. For a path of length L with 8-connectivity, that's ~8L expansions. The goal-side expansion is computed once and reused. This makes path reconstruction the most expensive query operation — significantly slower than a single distance query.

---

## 9. Pipeline Orchestration

**File:** `pathcircuit/core/pipeline.py`

The `build_pathcircuit_oracle(grid)` function ties all phases together:

```python
def build_pathcircuit_oracle(grid):
    padded = grid.to_padded()
    N = padded.rows

    # Phase 1: compute all base blocks
    base_blocks = [[compute_base_block(br, bc, padded)
                     for bc in range(N // 4)]
                    for br in range(N // 4)]

    # Special case: single 4×4 block needs no hierarchy
    if N == 4:
        bb = base_blocks[0][0]
        bb.D_global = bb.T.copy()
        bb.D_global_cells = list(bb.boundary_cells)
        bb.D_global_cell_to_index = dict(bb.boundary_cell_to_index)
        return Oracle(base_blocks=base_blocks, N=N, grid=padded)

    # Phase 2: hierarchical composition
    root, all_nodes = build_hierarchy(base_blocks, N, padded)

    # Phase 3a: un-condensation
    un_condense_all(root, N)

    # Phase 3b: build oracle with ancestor chains
    return build_oracle(root, base_blocks, N, padded)
```

The single-block special case handles 4×4 grids that have no hierarchy: the base block's D matrix already contains all-pairs shortest paths, so no merging or un-condensation is needed.

---

## 10. Design Decisions and Trade-offs

### Why 4×4 Base Blocks?

The block size of 4 is the sweet spot:
- **Small enough** that APSP by repeated squaring (4 rounds of 16×16 matmul) is cheap.
- **Large enough** that the ratio of interior to boundary cells (4:12) is meaningful — the Schur complement actually eliminates vertices and compresses information.
- **Powers of 2** make the quadtree structure clean: each base block is 4×4, each level-0 merge produces 8×8, then 16×16, etc.

### Why Store Both M_ES and M_ES×M_SS*?

The un-condensation formula for `D_SS` requires `M_SE ⊗ D_EE ⊗ M_ES` (using `M_ES`, not `M_ES × M_SS*`). The precomputed product `M_ES × M_SS*` is used for `D_ES`. Both are needed, so both are stored on the MergeNode.

### Why Greedy Path Reconstruction Instead of Witness-Based?

The witness matrices from repeated squaring and Kleene closure could theoretically enable O(L log n) path reconstruction via recursive bisection. However:
- The implementation complexity is significantly higher (multi-level witness chaining across the hierarchy).
- Greedy reconstruction is simple, correct by construction, and sufficient for the grid sizes targeted (up to 64×64).
- The witness matrices are still stored in case this optimisation is added later.

### Why the Padded Grid Is Used Everywhere (Not the Original)

After padding, the padded grid is passed to all phases. This means:
- The oracle stores distances in padded coordinates.
- Queries use padded coordinates internally.
- The UI translates between original and padded coordinates when interacting with the oracle.

Using a single coordinate system throughout avoids off-by-one errors and simplifies the boundary enumeration logic. Padding cells are SENTINEL-cost obstacles, so they never participate in any shortest path.

### Why SENTINEL = 2²⁹?

The value must be large enough to represent "infinity" (unreachable) but small enough that `SENTINEL + SENTINEL` doesn't overflow int32 (max 2³¹ − 1). Since 2²⁹ + 2²⁹ = 2³⁰ < 2³¹ − 1, this satisfies both constraints. It also leaves room for legitimate large distances: a 64×64 grid with all costs = 1 has a maximum shortest-path distance of about 2 × 128 = 256, far below SENTINEL.
