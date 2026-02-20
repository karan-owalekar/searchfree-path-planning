# Path Circuits: Complete Implementation Specification v2

This document specifies every data structure, index mapping, operation, and edge case needed to implement the Path Circuit algorithm. The target is a CPU reference implementation that takes an N×N grid with obstacles and costs, builds the hierarchical oracle, answers distance queries, and reconstructs cell-by-cell shortest paths. The output must be bitwise identical to Dijkstra for all pairs.

This is a corrected version addressing issues found in the v1 spec. Key changes are marked with **[V2 FIX]** throughout.

---

## 1. Constants and Configuration

```
BASE_BLOCK_SIZE = 4
BASE_CELLS = 16
BASE_BOUNDARY_COUNT = 12
BASE_INTERIOR_COUNT = 4
SENTINEL = 536870912             # 2^29
CONNECTIVITY = 4
```

**Integer type:** Signed 32-bit integers (Int32). Max representable: 2,147,483,647.

**Tropical addition with clamping:**
```
function tropical_add(a, b):
    if a >= SENTINEL or b >= SENTINEL:
        return SENTINEL
    sum = a + b
    if sum > SENTINEL:
        return SENTINEL
    if sum < 0:
        return SENTINEL          # overflow — should not happen with valid inputs
    return sum
```

**[V2 FIX]** Added early-exit when either operand is SENTINEL. This avoids computing a+b when one input is already infinity, preventing any possibility of overflow before the clamp check.

**Tropical min:**
```
function tropical_min(a, b):
    if a <= b: return a
    else: return b
```

---

## 2. Input Specification

**Input:** An N×N grid where N is divisible by 4. If not, pad:
```
function pad_grid(grid_costs, N):
    N_padded = ceil(N / 4) * 4
    padded = N_padded × N_padded matrix filled with SENTINEL
    for r = 0 to N-1:
        for c = 0 to N-1:
            padded[r][c] = grid_costs[r][c]
    return padded, N_padded
```

Each cell cost: `cost[row][col] ∈ {0, 1, 2, ..., MAX_COST} ∪ {SENTINEL}`

**Edge weight:**
```
function edge_weight(cell_i, cell_j, grid_costs):
    ci = grid_costs[cell_i.row][cell_i.col]
    cj = grid_costs[cell_j.row][cell_j.col]
    if ci >= SENTINEL or cj >= SENTINEL:
        return SENTINEL
    return ci + cj
```

**Grid coordinate to global cell index:**
```
function global_index(row, col, N):
    return row * N + col
```

---

## 3. Grid Partitioning

### 3.1 Block Grid

The N×N grid is divided into (N/4) × (N/4) base blocks. Block (br, bc) covers:
```
rows: [br * 4, br * 4 + 3]
cols: [bc * 4, bc * 4 + 3]
```

### 3.2 Local Indexing

The 16 cells within a block are labeled 0–15 in row-major order:
```
 0  1  2  3       B  B  B  B
 4  5  6  7       B  ι  ι  B
 8  9 10 11       B  ι  ι  B
12 13 14 15       B  B  B  B

local_row(idx) = idx / 4    (integer division)
local_col(idx) = idx % 4
global_row(br, idx) = br * 4 + local_row(idx)
global_col(bc, idx) = bc * 4 + local_col(idx)
```

### 3.3 Boundary and Interior Sets

```
BOUNDARY_INDICES = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
INTERIOR_INDICES = [5, 6, 9, 10]

boundary_to_local = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]

local_to_boundary = {
    0:0, 1:1, 2:2, 3:3, 4:4, 7:5, 8:6, 11:7, 12:8, 13:9, 14:10, 15:11
}
# Interior cells 5, 6, 9, 10 have no boundary index.
```

### 3.4 4-Neighbor Adjacency Within a 4×4 Block

```
NEIGHBORS_4x4 = {
    0:  [1, 4],         1:  [0, 2, 5],      2:  [1, 3, 6],      3:  [2, 7],
    4:  [0, 5, 8],      5:  [1, 4, 6, 9],   6:  [2, 5, 7, 10],  7:  [3, 6, 11],
    8:  [4, 9, 12],     9:  [5, 8, 10, 13],  10: [6, 9, 11, 14], 11: [7, 10, 15],
    12: [8, 13],        13: [9, 12, 14],     14: [10, 13, 15],    15: [11, 14]
}
```

---

## 4. Phase 1: Base Block Computation

### 4.1 Build the 16×16 Weight Matrix

```
function build_weight_matrix(br, bc, grid_costs):
    W = 16×16 matrix filled with SENTINEL

    for i = 0 to 15:
        W[i][i] = 0

        ri = br * 4 + i / 4
        ci = bc * 4 + i % 4
        cost_i = grid_costs[ri][ci]

        for j in NEIGHBORS_4x4[i]:
            rj = br * 4 + j / 4
            cj = bc * 4 + j % 4
            cost_j = grid_costs[rj][cj]

            if cost_i >= SENTINEL or cost_j >= SENTINEL:
                W[i][j] = SENTINEL
            else:
                W[i][j] = cost_i + cost_j

    return W
```

### 4.2 Tropical Matrix Multiplication

```
function tropical_matmul(A, B, n, m, p):
    C = n×p matrix filled with SENTINEL
    for i = 0 to n-1:
        for j = 0 to p-1:
            for k = 0 to m-1:
                val = tropical_add(A[i][k], B[k][j])
                C[i][j] = tropical_min(C[i][j], val)
    return C
```

### 4.3 Tropical Matrix Multiplication with Witness Recording

```
function tropical_matmul_with_witness(A, B, n, m, p):
    C = n×p matrix filled with SENTINEL
    W = n×p matrix filled with -1

    for i = 0 to n-1:
        for j = 0 to p-1:
            for k = 0 to m-1:
                val = tropical_add(A[i][k], B[k][j])
                if val < C[i][j]:
                    C[i][j] = val
                    W[i][j] = k
    return C, W
```

### 4.4 Compute All-Pairs Distance Matrix and Witnesses

```
function compute_base_block(br, bc, grid_costs):
    W = build_weight_matrix(br, bc, grid_costs)

    S1, w1 = tropical_matmul_with_witness(W, W, 16, 16, 16)     # W^2
    S2, w2 = tropical_matmul_with_witness(S1, S1, 16, 16, 16)   # W^4
    S3, w3 = tropical_matmul_with_witness(S2, S2, 16, 16, 16)   # W^8
    D,  w4 = tropical_matmul_with_witness(S3, S3, 16, 16, 16)   # W^16

    # Extract 12×12 transfer matrix
    T = 12×12 matrix
    for a = 0 to 11:
        for b = 0 to 11:
            T[a][b] = D[BOUNDARY_INDICES[a]][BOUNDARY_INDICES[b]]

    # Build boundary cell global coordinates and lookup
    boundary_cells = []
    boundary_cell_to_index = {}
    for bi = 0 to 11:
        local = BOUNDARY_INDICES[bi]
        r = br * 4 + local / 4
        c = bc * 4 + local % 4
        boundary_cells.append((r, c))
        boundary_cell_to_index[(r, c)] = bi

    return BaseBlock {
        block_row: br,
        block_col: bc,
        D: D,
        T: T,
        witnesses: [w1, w2, w3, w4],
        boundary_cells: boundary_cells,
        boundary_cell_to_index: boundary_cell_to_index,
        region_top: br * 4,
        region_left: bc * 4,
        region_height: 4,
        region_width: 4
    }
```

### 4.5 Data Structure

```
struct BaseBlock:
    block_row: int
    block_col: int
    D: int[16][16]
    T: int[12][12]
    witnesses: int[4][16][16]
    boundary_cells: list of (row, col)       # 12 cells, ordered
    boundary_cell_to_index: dict (row,col)→int
    region_top, region_left: int
    region_height, region_width: int         # always 4, 4
```

---

## 5. Hierarchy Data Structures

### 5.1 Hierarchy Node

**[V2 FIX]** Each binary merge (horizontal or vertical) produces one MergeNode. A 2×2 composition produces three MergeNodes: two from horizontal merges, one from the vertical merge. The vertical merge's node is the "level-k block" used by the next level.

```
struct MergeNode:
    level: int
    merge_type: "horizontal" or "vertical"

    # Region covered
    region_top, region_left: int
    region_height, region_width: int

    # Boundary of the combined block
    boundary_size: int                       # |E|
    boundary_cells: list of (row, col)
    boundary_cell_to_index: dict (row,col)→int

    # Transfer matrix
    T: int[boundary_size][boundary_size]

    # Schur complement data
    interface_size: int                      # |S|
    interface_cells: list of (row, col)
    M_SS_star: int[interface_size][interface_size]
    M_ES_times_MSS_star: int[boundary_size][interface_size]
    M_ES: int[boundary_size][interface_size]      # [V2 FIX] stored separately
    M_SE: int[interface_size][boundary_size]

    # Witnesses from Kleene closure
    kleene_witnesses: list of int[interface_size][interface_size]

    # Children (the two blocks that were merged)
    child_left: MergeNode or BaseBlock       # A for horizontal, AB for vertical
    child_right: MergeNode or BaseBlock      # B for horizontal, CD for vertical

    # Un-condensation result (filled in Phase 3a)
    D_global: int[boundary_size + interface_size][boundary_size + interface_size]
    D_global_cells: list of (row, col)       # ordering: boundary_cells + interface_cells
    D_global_cell_to_index: dict (row,col)→int
```

### 5.2 Boundary Ordering Convention

For any rectangular block covering rows [r_top, r_bot] and cols [c_left, c_right]:

```
function enumerate_boundary(r_top, r_bot, c_left, c_right):
    cells = []
    # Top edge (left to right)
    for c = c_left to c_right:
        cells.append((r_top, c))
    # Right edge (top+1 to bot-1, excluding corners)
    for r = r_top + 1 to r_bot - 1:
        cells.append((r, c_right))
    # Bottom edge (left to right)
    for c = c_left to c_right:
        cells.append((r_bot, c))
    # Left edge (top+1 to bot-1, excluding corners)
    for r = r_top + 1 to r_bot - 1:
        cells.append((r, c_left))
    return cells
```

**Boundary size:** For height h, width w: `b = 2*h + 2*w - 4`.

**Helper:**
```
function are_grid_adjacent(cell1, cell2):
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1]) == 1
```

---

## 6. Phase 2: Hierarchical Composition

### 6.1 Level Structure

```
L = log2(N / 4)         # total composition levels

Level 0: merge 4×4 blocks → 8×8 blocks
Level 1: merge 8×8 blocks → 16×16 blocks
...
Level L-1: final merge → N×N block
```

### 6.2 The 2×2 Grouping

```
At each level, four sub-blocks:
    [A] [B]       Merge 1 (H): A ∥ B → AB
    [C] [D]       Merge 2 (H): C ∥ D → CD   (parallel with Merge 1)
                  Merge 3 (V): AB ⫽ CD → ABCD
```

**[V2 FIX]** All three MergeNodes (AB, CD, ABCD) are stored in the hierarchy. Un-condensation needs all three.

### 6.3 The Horizontal Merge Operator

**Inputs:**
- Block A (left): boundary_cells, boundary_cell_to_index, T, region bounds
- Block B (right): boundary_cells, boundary_cell_to_index, T, region bounds
- grid_costs (for cross-edge weights)
- Preconditions: A.region_left + A.region_width == B.region_left; same row span

**Step 1: Identify interface and external cells.**

```
function identify_horizontal_partition(A, B):
    shared_col_A = A.region_left + A.region_width - 1     # A's rightmost col
    shared_col_B = B.region_left                           # B's leftmost col
    r_top = A.region_top
    r_bot = A.region_top + A.region_height - 1

    # Corner cells (stay external):
    corners_A = {(r_top, shared_col_A), (r_bot, shared_col_A)}
    corners_B = {(r_top, shared_col_B), (r_bot, shared_col_B)}

    # Interface cells (to be eliminated):
    I_A = [(r, shared_col_A) for r in range(r_top + 1, r_bot)]
    I_B = [(r, shared_col_B) for r in range(r_top + 1, r_bot)]
    S = I_A + I_B

    # External cells
    E_A = [c for c in A.boundary_cells if c not in set(I_A)]
    E_B = [c for c in B.boundary_cells if c not in set(I_B)]

    # Combined block boundary in canonical order
    c_left = A.region_left
    c_right = B.region_left + B.region_width - 1
    C_boundary = enumerate_boundary(r_top, r_bot, c_left, c_right)

    return C_boundary, S, I_A, I_B, E_A, E_B
```

**Step 2: Build index mappings.**

```
function build_merge_mappings(C_boundary, S, A, B):
    # Map each external cell to its source block and index
    E_map = []     # list of (source='A'|'B', index_in_source_T)
    for cell in C_boundary:
        if cell in A.boundary_cell_to_index:
            E_map.append(('A', A.boundary_cell_to_index[cell]))
        elif cell in B.boundary_cell_to_index:
            E_map.append(('B', B.boundary_cell_to_index[cell]))
        else:
            ERROR("Cell not in either boundary: " + str(cell))

    # Map each interface cell to its source
    S_A_map = [A.boundary_cell_to_index[c] for c in I_A]
    S_B_map = [B.boundary_cell_to_index[c] for c in I_B]

    return E_map, S_A_map, S_B_map
```

**Step 3: Assemble the joint matrix M.**

```
function assemble_joint_matrix(T_A, T_B, E_map, S_A_map, S_B_map,
                               I_A, I_B, C_boundary, grid_costs):
    nE = len(E_map)
    nS = len(S_A_map) + len(S_B_map)
    n = nE + nS
    M = n × n matrix filled with SENTINEL

    # --- M_EE: E×E block (rows 0..nE-1, cols 0..nE-1) ---
    for i = 0 to nE - 1:
        for j = 0 to nE - 1:
            si, idx_i = E_map[i]
            sj, idx_j = E_map[j]
            if si == sj:
                T = T_A if si == 'A' else T_B
                M[i][j] = T[idx_i][idx_j]
            else:
                # Cross-block: only grid-adjacent corner cells
                if are_grid_adjacent(C_boundary[i], C_boundary[j]):
                    M[i][j] = edge_weight(C_boundary[i], C_boundary[j], grid_costs)
                # else remains SENTINEL

    # --- M_ES and M_SE: E×S and S×E blocks ---
    for i = 0 to nE - 1:
        si, idx_i = E_map[i]
        for j = 0 to nS - 1:
            sj_idx = nE + j
            if j < len(S_A_map):
                # S cell from I_A (block A)
                if si == 'A':
                    M[i][sj_idx] = T_A[idx_i][S_A_map[j]]
                    M[sj_idx][i] = T_A[S_A_map[j]][idx_i]
                # else SENTINEL (different blocks, no direct connection)
            else:
                # S cell from I_B (block B)
                jB = j - len(S_A_map)
                if si == 'B':
                    M[i][sj_idx] = T_B[idx_i][S_B_map[jB]]
                    M[sj_idx][i] = T_B[S_B_map[jB]][idx_i]

    # --- M_SS: S×S block ---
    for i = 0 to nS - 1:
        for j = 0 to nS - 1:
            si = nE + i
            sj = nE + j
            inA_i = i < len(S_A_map)
            inA_j = j < len(S_A_map)

            if inA_i and inA_j:
                M[si][sj] = T_A[S_A_map[i]][S_A_map[j]]
            elif (not inA_i) and (not inA_j):
                iB = i - len(S_A_map)
                jB = j - len(S_A_map)
                M[si][sj] = T_B[S_B_map[iB]][S_B_map[jB]]
            else:
                # Cross-block: check same-row adjacency
                if inA_i:
                    cell_a = I_A[i]
                    cell_b = I_B[j - len(S_A_map)]
                else:
                    cell_a = I_A[j]
                    cell_b = I_B[i - len(S_A_map)]
                if cell_a[0] == cell_b[0]:   # same row → grid adjacent
                    w = edge_weight(cell_a, cell_b, grid_costs)
                    M[si][sj] = w
                    M[sj][si] = w

    return M
```

**Step 4: Extract sub-matrices.**

```
M_EE = M[0:nE, 0:nE]
M_ES = M[0:nE, nE:nE+nS]
M_SE = M[nE:nE+nS, 0:nE]
M_SS = M[nE:nE+nS, nE:nE+nS]
```

**Step 5: Compute Kleene closure M_SS*.**

**[V2 FIX]** Removed the redundant accumulation line. With zero diagonals, tropical matrix powers are monotonically non-increasing, so R^(2^t) already captures all walks of length ≤ 2^t. No element-wise min with the previous iterate is needed.

```
function kleene_closure(M_SS, s_size):
    num_squarings = max(1, ceil(log2(s_size)))

    R = copy(M_SS)
    witnesses = []

    for t = 1 to num_squarings:
        R_new, W_t = tropical_matmul_with_witness(R, R, s_size, s_size, s_size)
        R = R_new
        witnesses.append(W_t)

    return R, witnesses
```

**Why no accumulation is needed:** R starts as M_SS with R[i][i] = 0 for all i. After the first squaring, R_new = R ⊗ R. For any i,j: R_new[i][j] = min_k(R[i][k] + R[k][j]). When k = i: R[i][i] + R[i][j] = 0 + R[i][j] = R[i][j]. So R_new[i][j] ≤ R[i][j] always. The new matrix is entry-wise ≤ the old one. The old values are automatically included. This holds at every subsequent squaring step. After ⌈log₂|S|⌉ squarings, R = M_SS^(2^⌈log₂|S|⌉), which has stabilized (equals M_SS* since 2^⌈log₂|S|⌉ ≥ |S| and the longest simple path has |S|−1 edges).

**Why witnesses are correct without accumulation:** Since R_new[i][j] ≤ R[i][j] always, the minimum at each step is always achieved by the matmul (not by keeping the old value). The witness W_t[i][j] records which k achieved this minimum, which is the correct midpoint for recursive bisection.

**Step 6: Compute composed transfer matrix.**

```
# Precompute products needed for queries and un-condensation
M_ES_times_MSS_star = tropical_matmul(M_ES, M_SS_star, nE, nS, nS)
schur_product = tropical_matmul(M_ES_times_MSS_star, M_SE, nE, nS, nE)

# T_C = M_EE ⊕ schur_product
T_C = nE × nE matrix
for i = 0 to nE - 1:
    for j = 0 to nE - 1:
        T_C[i][j] = tropical_min(M_EE[i][j], schur_product[i][j])
```

**Step 7: Build and store the MergeNode.**

```
C_boundary_to_index = {cell: i for i, cell in enumerate(C_boundary)}

node = MergeNode {
    level: level,
    merge_type: "horizontal",
    region_top: A.region_top,
    region_left: A.region_left,
    region_height: A.region_height,
    region_width: A.region_width + B.region_width,
    boundary_size: nE,
    boundary_cells: C_boundary,
    boundary_cell_to_index: C_boundary_to_index,
    T: T_C,
    interface_size: nS,
    interface_cells: S,
    M_SS_star: M_SS_star,
    M_ES_times_MSS_star: M_ES_times_MSS_star,
    M_ES: M_ES,                          # [V2 FIX] stored for un-condensation
    M_SE: M_SE,
    kleene_witnesses: witnesses,
    child_left: A,
    child_right: B,
    D_global: None,
    D_global_cells: None,
    D_global_cell_to_index: None
}
return node
```

### 6.4 The Vertical Merge Operator

Identical logic with these substitutions:

```
function identify_vertical_partition(A, B):
    # A is top, B is bottom
    shared_row_A = A.region_top + A.region_height - 1    # A's bottom row
    shared_row_B = B.region_top                           # B's top row
    c_left = A.region_left
    c_right = A.region_left + A.region_width - 1

    # Corners (stay external): leftmost and rightmost of each shared row
    corners_A = {(shared_row_A, c_left), (shared_row_A, c_right)}
    corners_B = {(shared_row_B, c_left), (shared_row_B, c_right)}

    # Interface cells
    I_A = [(shared_row_A, c) for c in range(c_left + 1, c_right)]
    I_B = [(shared_row_B, c) for c in range(c_left + 1, c_right)]
    S = I_A + I_B

    E_A = [c for c in A.boundary_cells if c not in set(I_A)]
    E_B = [c for c in B.boundary_cells if c not in set(I_B)]

    r_top = A.region_top
    r_bot = B.region_top + B.region_height - 1
    C_boundary = enumerate_boundary(r_top, r_bot, c_left, c_right)

    return C_boundary, S, I_A, I_B, E_A, E_B
```

All subsequent steps (mapping, assembly, Kleene closure, Schur complement, storage) are identical to the horizontal case. Set `merge_type: "vertical"`.

### 6.5 Full Phase 2 Procedure

**[V2 FIX]** All three MergeNodes per group are stored. The vertical merge node becomes the "block" for the next level. The horizontal merge nodes (AB, CD) are stored inside the vertical merge node for un-condensation access.

```
function build_hierarchy(base_blocks, N, grid_costs):
    L = log2(N / 4)

    # Wrap base blocks (they already have boundary_cells etc from §4.4)
    prev_level = 2D array of base_blocks    # (N/4) × (N/4)

    all_merge_nodes = []    # flat list for un-condensation ordering

    for level = 0 to L - 1:
        rows_prev = num_rows(prev_level)
        cols_prev = num_cols(prev_level)
        rows_next = rows_prev / 2
        cols_next = cols_prev / 2

        next_level = 2D array of rows_next × cols_next

        for gr = 0 to rows_next - 1:
            for gc = 0 to cols_next - 1:
                A = prev_level[2*gr][2*gc]
                B = prev_level[2*gr][2*gc + 1]
                C = prev_level[2*gr + 1][2*gc]
                D = prev_level[2*gr + 1][2*gc + 1]

                AB = horizontal_merge(A, B, level, grid_costs)
                CD = horizontal_merge(C, D, level, grid_costs)
                ABCD = vertical_merge(AB, CD, level, grid_costs)

                # Store H-merge nodes inside the V-merge node
                ABCD.h_merge_left = AB
                ABCD.h_merge_right = CD

                next_level[gr][gc] = ABCD
                all_merge_nodes.append(ABCD)
                all_merge_nodes.append(AB)
                all_merge_nodes.append(CD)

        prev_level = next_level

    root = prev_level[0][0]
    return root, all_merge_nodes
```

---

## 7. Phase 3a: Un-Condensation (Top-Down)

### 7.1 Purpose

Phase 2 eliminates interface vertices at each merge. Un-condensation recovers global distances involving those eliminated vertices, enabling O(log N) queries.

### 7.2 Processing Order

**[V2 FIX]** Each 2×2 group has three MergeNodes. Un-condensation processes them in reverse merge order: vertical first (ABCD), then both horizontal (AB, CD). This is done level by level, top-down.

```
function un_condense_all(root, L):
    # Initialize root
    root.D_global = root.T
    root.D_global_cells = list(root.boundary_cells)
    root.D_global_cell_to_index = dict(root.boundary_cell_to_index)

    # Process levels top-down
    for level = L - 1 down to 0:
        # Collect all ABCD nodes at this level
        # (In practice, traverse the tree or use the stored list)
        for each ABCD node at this level:

            # Step A: Un-condense the vertical merge (ABCD)
            un_condense_single_merge(ABCD)

            # Step B: Propagate D_global to AB and CD
            propagate_D_global(ABCD, ABCD.h_merge_left)   # → AB
            propagate_D_global(ABCD, ABCD.h_merge_right)  # → CD

            # Step C: Un-condense the horizontal merges
            un_condense_single_merge(ABCD.h_merge_left)    # AB
            un_condense_single_merge(ABCD.h_merge_right)   # CD

            # Step D: Propagate D_global to the four leaf children
            AB = ABCD.h_merge_left
            CD = ABCD.h_merge_right
            propagate_D_global(AB, AB.child_left)    # → A
            propagate_D_global(AB, AB.child_right)   # → B
            propagate_D_global(CD, CD.child_left)    # → C
            propagate_D_global(CD, CD.child_right)   # → D
```

### 7.3 Un-Condensation at a Single Merge Node

```
function un_condense_single_merge(node):
    nE = node.boundary_size
    nS = node.interface_size

    if nS == 0:
        return    # nothing to un-condense

    D_EE = node.D_global    # nE × nE, already set by parent propagation

    # Stored from Phase 2:
    M_ES_MSS = node.M_ES_times_MSS_star    # nE × nS
    M_ES = node.M_ES                        # nE × nS  [V2 FIX]
    M_SE = node.M_SE                        # nS × nE
    M_SS_star = node.M_SS_star              # nS × nS

    # D_ES: distances from E to S
    # D_ES[e][s] = min_{e'} (D_EE[e][e'] + M_ES_MSS[e'][s])
    D_ES = tropical_matmul(D_EE, M_ES_MSS, nE, nE, nS)

    # D_SE: transpose (symmetric distances)
    D_SE = transpose(D_ES, nE, nS)

    # D_SS: distances among S vertices
    # D_SS = M_SS_star ⊕ (M_SE ⊗ D_EE ⊗ M_ES)
    temp = tropical_matmul(M_SE, D_EE, nS, nE, nE)
    temp2 = tropical_matmul(temp, M_ES, nS, nE, nS)    # [V2 FIX] uses M_ES, not M_ES_MSS

    D_SS = nS × nS matrix
    for i = 0 to nS - 1:
        for j = 0 to nS - 1:
            D_SS[i][j] = tropical_min(M_SS_star[i][j], temp2[i][j])

    # Assemble full distance matrix
    total = nE + nS
    D_full = total × total matrix
    D_full[0:nE, 0:nE] = D_EE
    D_full[0:nE, nE:total] = D_ES
    D_full[nE:total, 0:nE] = D_SE
    D_full[nE:total, nE:total] = D_SS

    node.D_global = D_full
    node.D_global_cells = list(node.boundary_cells) + list(node.interface_cells)
    node.D_global_cell_to_index = {cell: i for i, cell in enumerate(node.D_global_cells)}
```

**Helper:**
```
function transpose(M, rows, cols):
    MT = cols × rows matrix
    for i = 0 to rows - 1:
        for j = 0 to cols - 1:
            MT[j][i] = M[i][j]
    return MT
```

### 7.4 Propagating D_global to Children

**[V2 FIX]** This is the key function connecting levels. After un-condensing a merge node, its D_global contains distances among all E ∪ S cells. Each child's boundary cells are a subset of these E ∪ S cells (the child contributed them during the merge). Extract the relevant sub-matrix.

```
function propagate_D_global(parent, child):
    if child is a BaseBlock:
        # Base blocks don't need D_global propagation —
        # they already have full 16×16 D matrices.
        # But we DO need to set their D_global for the oracle index.
        # Their boundary cells should all appear in parent's D_global.
        indices = []
        for cell in child.boundary_cells:
            if cell in parent.D_global_cell_to_index:
                indices.append(parent.D_global_cell_to_index[cell])
            else:
                ERROR("Child boundary cell not in parent D_global: " + str(cell))

        n = len(indices)
        child_D_global = n × n matrix
        for i = 0 to n - 1:
            for j = 0 to n - 1:
                child_D_global[i][j] = parent.D_global[indices[i]][indices[j]]

        child.D_global = child_D_global
        child.D_global_cells = list(child.boundary_cells)
        child.D_global_cell_to_index = dict(child.boundary_cell_to_index)
        return

    # For MergeNode children:
    # The child's boundary cells are a subset of parent's E ∪ S cells
    indices = []
    for cell in child.boundary_cells:
        if cell in parent.D_global_cell_to_index:
            indices.append(parent.D_global_cell_to_index[cell])
        else:
            ERROR("Child boundary cell not in parent D_global: " + str(cell))

    n = len(indices)
    child_D_EE = n × n matrix
    for i = 0 to n - 1:
        for j = 0 to n - 1:
            child_D_EE[i][j] = parent.D_global[indices[i]][indices[j]]

    child.D_global = child_D_EE
    # D_global_cells will be updated after child's own un-condensation
```

---

## 8. Phase 3b: Querying

### 8.1 Oracle Index

After un-condensation, build a lookup structure:

```
struct Oracle:
    base_blocks: (N/4) × (N/4) array of BaseBlock
    # For each grid cell that is a boundary/interface cell,
    # the lowest-level node where it appears in D_global
    cell_lookup: dict (row,col) → (node, index_in_D_global)

function build_oracle(root, base_blocks):
    oracle = Oracle { base_blocks: base_blocks, cell_lookup: {} }

    # Walk the hierarchy. For each cell, record the LOWEST-level node
    # where it appears. Lower = more specific = better.
    populate_cell_lookup(root, oracle)

    return oracle

function populate_cell_lookup(node, oracle):
    if node is None:
        return

    # Register all cells in this node's D_global
    for i, cell in enumerate(node.D_global_cells):
        if cell not in oracle.cell_lookup:
            oracle.cell_lookup[cell] = (node, i)
        else:
            # Keep the lower-level (more specific) entry
            existing_node, _ = oracle.cell_lookup[cell]
            if node.level < existing_node.level:
                oracle.cell_lookup[cell] = (node, i)

    # Recurse into children
    if node is MergeNode:
        populate_cell_lookup(node.child_left, oracle)
        populate_cell_lookup(node.child_right, oracle)
        if hasattr(node, 'h_merge_left'):
            populate_cell_lookup(node.h_merge_left, oracle)
            populate_cell_lookup(node.h_merge_right, oracle)
```

**[V2 FIX]** For the distance lookup between two boundary cells b1 and b2, we need the lowest node whose D_global contains BOTH cells. The cell_lookup gives us where each cell lives, but we need a common node. Build a pairwise lookup:

```
function build_pairwise_lookup(oracle):
    # For each pair of boundary cells that might be queried,
    # find the lowest common node.
    # In practice: walk up from each cell's lowest node to find common ancestor.
    # This is precomputable but O(N² log N) — do it lazily instead.
    pass

function find_common_distance(oracle, cell1, cell2):
    # Strategy: check nodes that contain cell1, from lowest level up,
    # until we find one that also contains cell2.

    # Get all nodes containing cell1 (walk up the tree from cell1's lowest node)
    node1, idx1 = oracle.cell_lookup[cell1]

    # Check if cell2 is in the same node
    if cell2 in node1.D_global_cell_to_index:
        idx2 = node1.D_global_cell_to_index[cell2]
        return node1.D_global[idx1][idx2]

    # Walk up: the parent of node1 should contain cell1 and possibly cell2
    # We need a parent pointer or to search the tree.
    # SIMPLER APPROACH: iterate over all nodes that contain cell1
    # from lowest to highest level, check for cell2.

    # For the reference implementation, use a brute-force search:
    result = SENTINEL
    for node in all_nodes_sorted_by_level_ascending:
        if cell1 in node.D_global_cell_to_index and cell2 in node.D_global_cell_to_index:
            i1 = node.D_global_cell_to_index[cell1]
            i2 = node.D_global_cell_to_index[cell2]
            result = tropical_min(result, node.D_global[i1][i2])
            break    # lowest level with both cells gives the best answer
    return result
```

**Optimization:** Add parent pointers to each node, then walk up from the lower-level node until both cells are found. This makes lookup O(log N).

### 8.2 Distance Query

```
function query_distance(oracle, s, g):
    s_row, s_col = s
    g_row, g_col = g

    # Same cell
    if s == g:
        return 0

    s_br = s_row / 4
    s_bc = s_col / 4
    g_br = g_row / 4
    g_bc = g_col / 4

    # Same base block
    if s_br == g_br and s_bc == g_bc:
        bb = oracle.base_blocks[s_br][s_bc]
        s_local = (s_row % 4) * 4 + (s_col % 4)
        g_local = (g_row % 4) * 4 + (g_col % 4)
        return bb.D[s_local][g_local]

    # Different base blocks: minimize over boundary pairs
    bb_s = oracle.base_blocks[s_br][s_bc]
    bb_g = oracle.base_blocks[g_br][g_bc]
    s_local = (s_row % 4) * 4 + (s_col % 4)
    g_local = (g_row % 4) * 4 + (g_col % 4)

    best = SENTINEL

    for a = 0 to 11:
        d_s_bs = bb_s.D[s_local][BOUNDARY_INDICES[a]]
        if d_s_bs >= SENTINEL:
            continue

        bs_cell = bb_s.boundary_cells[a]

        for b = 0 to 11:
            d_bg_g = bb_g.D[BOUNDARY_INDICES[b]][g_local]
            if d_bg_g >= SENTINEL:
                continue

            bg_cell = bb_g.boundary_cells[b]

            d_mid = find_common_distance(oracle, bs_cell, bg_cell)

            total = tropical_add(d_s_bs, tropical_add(d_mid, d_bg_g))
            best = tropical_min(best, total)

    return best
```

---

## 9. Path Reconstruction

### 9.1 Strategy

**[V2 FIX]** The reference implementation uses a **greedy oracle-assisted approach** for path reconstruction, rather than witness-based hierarchy traversal. This is simpler to implement and verify. The witness-based approach is an optimization described separately in §9.4.

**Greedy approach:** Starting at s, repeatedly step to the neighbor that minimizes d(neighbor, g) + edge_weight(current, neighbor). This is correct because oracle distances are exact and edge weights are non-negative (so greedy on exact distances never makes a suboptimal choice).

### 9.2 Greedy Path Reconstruction

```
function reconstruct_path(oracle, s, g):
    if s == g:
        return [s]

    d_sg = query_distance(oracle, s, g)
    if d_sg >= SENTINEL:
        return []    # no path

    path = [s]
    current = s

    while current != g:
        best_next = None
        best_cost = SENTINEL

        for neighbor in grid_neighbors(current):
            w = edge_weight(current, neighbor, grid_costs)
            if w >= SENTINEL:
                continue
            d_ng = query_distance(oracle, neighbor, g)
            total = tropical_add(w, d_ng)
            if total < best_cost:
                best_cost = total
                best_next = neighbor

        if best_next is None:
            ERROR("Path reconstruction failed — should not happen with valid d(s,g)")

        path.append(best_next)
        current = best_next

    return path

function grid_neighbors(cell):
    r, c = cell
    result = []
    if r > 0:     result.append((r-1, c))
    if r < N-1:   result.append((r+1, c))
    if c > 0:     result.append((r, c-1))
    if c < N-1:   result.append((r, c+1))
    return result
```

**Cost of greedy reconstruction:** O(path_length × query_cost). With O(log N) oracle queries, this is O(path_length × log N). Each step queries at most 4 neighbors.

**Correctness:** At each step, the greedy choice selects the neighbor n* minimizing w(current, n*) + d(n*, g). Since d is exact, we have d(current, g) = w(current, n*) + d(n*, g) (the optimal next step always achieves this equality). So the greedy path has total cost d(s, g) — exactly optimal.

### 9.3 Path Validation

```
function validate_path(path, oracle, grid_costs):
    if len(path) < 2:
        return true

    total_cost = 0
    for i = 0 to len(path) - 2:
        w = edge_weight(path[i], path[i+1], grid_costs)
        if w >= SENTINEL:
            return false    # path crosses obstacle
        if not are_grid_adjacent(path[i], path[i+1]):
            return false    # non-adjacent step
        total_cost = tropical_add(total_cost, w)

    expected = query_distance(oracle, path[0], path[len(path)-1])
    return total_cost == expected
```

### 9.4 Witness-Based Reconstruction (Optimization, Not Required for Reference)

For higher performance, paths can be reconstructed using stored midpoint witnesses without per-step oracle queries. This requires:

1. Base block witnesses (stored in Phase 1): reconstruct cell-by-cell paths within 4×4 blocks via recursive bisection of the 4 squaring-step witness matrices.

2. Kleene closure witnesses (stored in Phase 2): reconstruct paths through interface vertices at each hierarchy level via recursive bisection.

3. Un-condensation witnesses (optionally stored in Phase 3a): reconstruct paths between external and interface vertices.

Implementation of witness-based reconstruction is deferred to the optimization phase. The greedy approach in §9.2 is correct, simpler, and sufficient for validation.

---

## 10. Validation Against Dijkstra

### 10.1 Reference Dijkstra

```
function dijkstra_sssp(grid_costs, N, source):
    dist = N×N matrix filled with SENTINEL
    s_r, s_c = source
    if grid_costs[s_r][s_c] >= SENTINEL:
        return dist    # source is obstacle
    dist[s_r][s_c] = 0

    heap = min-heap containing (0, source)

    while heap is not empty:
        (d, u) = heap.pop_min()
        if d > dist[u[0]][u[1]]:
            continue

        for v in grid_neighbors(u):
            w = edge_weight(u, v, grid_costs)
            if w >= SENTINEL:
                continue
            new_dist = tropical_add(d, w)
            if new_dist < dist[v[0]][v[1]]:
                dist[v[0]][v[1]] = new_dist
                heap.push((new_dist, v))

    return dist
```

### 10.2 Full Validation

```
function validate_oracle(oracle, grid_costs, N):
    errors = 0
    for s_r = 0 to N-1:
        for s_c = 0 to N-1:
            if grid_costs[s_r][s_c] >= SENTINEL:
                continue
            dijk = dijkstra_sssp(grid_costs, N, (s_r, s_c))
            for g_r = 0 to N-1:
                for g_c = 0 to N-1:
                    d_oracle = query_distance(oracle, (s_r,s_c), (g_r,g_c))
                    d_dijk = dijk[g_r][g_c]
                    if d_oracle != d_dijk:
                        print("MISMATCH ({s_r},{s_c})→({g_r},{g_c}): "
                              "oracle={d_oracle} dijk={d_dijk}")
                        errors += 1
    print("Errors: " + str(errors))
    return errors == 0
```

---

## 11. Numerical Trace: 4×4 Grid (Single Block, No Hierarchy)

This trace validates Phase 1 in isolation. A 4×4 grid is one base block — no composition needed.

### 11.1 Setup

```
Grid: 4×4, all costs = 1, no obstacles.

cost = [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]
```

### 11.2 Weight Matrix W

All adjacent pairs have weight 1 + 1 = 2. Diagonal = 0. Non-adjacent = SENTINEL.

```
W[0][1] = 2,  W[0][4] = 2                          (cell 0: neighbors 1, 4)
W[1][0] = 2,  W[1][2] = 2,  W[1][5] = 2            (cell 1: neighbors 0, 2, 5)
W[5][1] = 2,  W[5][4] = 2,  W[5][6] = 2,  W[5][9] = 2  (cell 5: 4 neighbors)
...etc for all cells per the adjacency table in §3.4
```

### 11.3 Expected D = W^16

Since all costs are uniform, D[i][j] = 2 × Manhattan_distance(i, j):

```
D[0][0]  = 0    D[0][1]  = 2    D[0][2]  = 4    D[0][3]  = 6
D[0][4]  = 2    D[0][5]  = 4    D[0][6]  = 6    D[0][7]  = 8
D[0][8]  = 4    D[0][9]  = 6    D[0][10] = 8    D[0][11] = 10
D[0][12] = 6    D[0][13] = 8    D[0][14] = 10   D[0][15] = 12
```

General formula: D[i][j] = 2 × (|row(i)−row(j)| + |col(i)−col(j)|)

### 11.4 Transfer Matrix T (12×12)

T[a][b] = D[β_a][β_b] where β = BOUNDARY_INDICES.

Selected entries:
```
T[0][0]  = D[0][0]   = 0     (cell 0 to cell 0)
T[0][3]  = D[0][3]   = 6     (cell 0 to cell 3, 3 hops)
T[0][11] = D[0][15]  = 12    (cell 0 to cell 15, 6 hops — maximum)
T[4][5]  = D[4][7]   = 6     (cell 4 to cell 7, 3 hops across middle row)
T[4][7]  = D[4][11]  = 4     (cell 4 to cell 11, 2 hops down right column)
```

### 11.5 Validation

Query d((0,0), (3,3)): same base block, read D[0][15] = 12. ✓
Dijkstra: path [0,0]→[0,1]→[0,2]→[0,3]→[1,3]→[2,3]→[3,3], cost = 2×6 = 12. ✓

---

## 12. Numerical Trace: 8×8 Grid (One Hierarchy Level)

### 12.1 Setup

```
8×8 grid, all costs = 1, one obstacle at (2, 5).

Block A: rows [0,3], cols [0,3] — uniform, no obstacles
Block B: rows [0,3], cols [4,7] — obstacle at local cell 9 (row 2, col 1 within B)
Block C: rows [4,7], cols [0,3] — uniform
Block D: rows [4,7], cols [4,7] — uniform
```

### 12.2 Phase 1: Block A

Same as §11. T_A[a][b] = 2 × Manhattan(β_a, β_b).

### 12.3 Phase 1: Block B

Block B local layout:
```
 0  1  2  3        costs: 1  1  1  1
 4  5  6  7               1  1  1  1
 8  9 10 11               1  ∞  1  1
12 13 14 15               1  1  1  1
```

W_B[9][anything] = SENTINEL, W_B[anything][9] = SENTINEL (except W_B[9][9] = 0).

After W_B^16, most entries match block A. Key differences:
```
D_B[8][10]:  Cannot go 8→9→10. Must detour: 8→4→5→6→10 (cost 8) or 8→12→13→14→10 (cost 8).
             So D_B[8][10] = 8 (vs 4 in block A).
```

T_B differs from T_A at entries involving paths through cell 9.

### 12.4 Phase 2: Horizontal Merge A ∥ B

```
A's right column: cells (0,3), (1,3), (2,3), (3,3) — local indices 3, 7, 11, 15
Corners: (0,3) and (3,3) — stay external
I_A: [(1,3), (2,3)] — local indices 7, 11 in A. |I_A| = 2.

B's left column: cells (0,4), (1,4), (2,4), (3,4) — local indices 0, 4, 8, 12
Corners: (0,4) and (3,4) — stay external
I_B: [(1,4), (2,4)] — local indices 4, 8 in B. |I_B| = 2.

S = [(1,3), (2,3), (1,4), (2,4)]. |S| = 4.
```

Cross-edges (same row):
```
c((1,3)) + c((1,4)) = 1 + 1 = 2
c((2,3)) + c((2,4)) = 1 + 1 = 2
```

M_SS is 4×4:
```
       (1,3)  (2,3)  (1,4)  (2,4)
(1,3) [  0      4      2      ∞  ]    T_A[7→7]=0, T_A[7→11]=4, cross=2
(2,3) [  4      0      ∞      2  ]    T_A[11→7]=4, T_A[11→11]=0, cross=2
(1,4) [  2      ∞      0      4  ]    cross=2, T_B[4→4]=0, T_B[4→8]=4
(2,4) [  ∞      2      4      0  ]    cross=2, T_B[8→4]=4, T_B[8→8]=0
```

Note: T_B[4→8] might differ from 4 if the obstacle affects that path. In block B, cell 4 (local row 1, col 0) to cell 8 (local row 2, col 0): path 4→0→... no, direct path is 4→8, cost = c(4)+c(8) = 2. The obstacle at cell 9 doesn't block this direct adjacency. So T_B[local 4, local 8] = D_B[4][8] = 2.

Corrected M_SS:
```
       (1,3)  (2,3)  (1,4)  (2,4)
(1,3) [  0      4      2      ∞  ]
(2,3) [  4      0      ∞      2  ]
(1,4) [  2      ∞      0      2  ]    T_B[4→8] = 2 (direct adjacency)
(2,4) [  ∞      2      2      0  ]    T_B[8→4] = 2
```

Kleene closure: ⌈log₂(4)⌉ = 2 squarings.

After 1st squaring (M_SS²):
```
M_SS²[0][2] = min(0+2, 4+∞, 2+0, ∞+2) = 2    (already same)
M_SS²[0][3] = min(0+∞, 4+2, 2+2, ∞+0) = 4    (was ∞, now 4 via path (1,3)→(2,3)→(2,4))
M_SS²[1][2] = min(4+2, 0+∞, ∞+0, 2+2) = 4    (was ∞, now 4)
```

After 2nd squaring: stabilized. M_SS* is the final matrix.

Schur complement produces T_AB: 20×20 transfer matrix for the 8×4 wide block.

### 12.5 Query Example

Query: s = (0,0), g = (2,6).

s is in block A (local cell 0). g is in block B (local cell 10).
s_local = 0, g_local = 10.

Base block distances:
```
bb_s.D[0][β_a] for a=0..11:  distances from cell (0,0) to each of A's 12 boundary cells
bb_g.D[β_b][10] for b=0..11: distances from each of B's boundary cells to cell (2,6)
```

The oracle minimizes over 144 pairs. The optimal path goes from (0,0) through A to some boundary cell of A, across through the hierarchy to some boundary cell of B, then through B (avoiding the obstacle) to (2,6).

Expected result: slightly more than 2 × Manhattan((0,0),(2,6)) = 16 due to the obstacle detour.

---

## 13. Edge Cases and Invariant Checks

### 13.1 Invariants (Assert After Every Matrix Computation)

```
function check_invariants(T, size, label):
    # Zero diagonal
    for i = 0 to size - 1:
        assert T[i][i] == 0, label + ": non-zero diagonal at " + str(i)

    # Symmetry
    for i = 0 to size - 1:
        for j = i + 1 to size - 1:
            assert T[i][j] == T[j][i], label + ": asymmetry at " + str(i) + "," + str(j)

    # Triangle inequality
    for i = 0 to size - 1:
        for j = 0 to size - 1:
            for k = 0 to size - 1:
                assert T[i][j] <= tropical_add(T[i][k], T[k][j]),
                    label + ": triangle violation"
```

Call `check_invariants` on:
- Every base block D (16×16) and T (12×12)
- Every M_SS_star
- Every composed T_C
- Every D_global after un-condensation

The triangle inequality check is O(n³) per matrix, so disable it for production after initial validation. Keep the diagonal and symmetry checks always.

### 13.2 Edge Cases

**All-obstacle block:** T[i][j] = SENTINEL for i ≠ j, T[i][i] = 0. Composition correctly propagates.

**Disconnected regions:** d(s,g) = SENTINEL. Oracle returns SENTINEL.

**Source or goal is obstacle:** D[s_local][anything] = SENTINEL (except self). Query returns SENTINEL.

**Zero-cost cells:** Valid. Edge weights can be 0. d(s,g) can be 0 only if s = g (since edge weights are c(i)+c(j) ≥ 0, and a non-trivial path has at least one edge with positive weight... unless both endpoints have cost 0 and are adjacent, giving edge weight 0. In that case d(s,g) = 0 for adjacent zero-cost cells. This is correct.)

**4×4 grid:** One base block, zero composition levels. Phase 2 is skipped. Queries read D directly.

**Grid not divisible by 4:** Pad with SENTINEL obstacle cells. Queries to padded cells return SENTINEL.

---

## 14. Memory Layout

### 14.1 Per Base Block

```
D: 16 × 16 × 4 bytes = 1,024 bytes
T: 12 × 12 × 4 bytes = 576 bytes
witnesses: 4 × 16 × 16 × 1 byte = 1,024 bytes
boundary_cells + index: ~200 bytes
Total: ~3 KB per block, N²/16 blocks → trivial
```

### 14.2 Per Merge Node at Level k

```
Block dims: variable (depends on H or V merge)
|E| = boundary_size, |S| = interface_size

T:                 |E|² × 4 bytes
M_SS_star:         |S|² × 4 bytes
M_ES_MSS_star:     |E|·|S| × 4 bytes
M_ES:              |E|·|S| × 4 bytes        [V2: now stored]
M_SE:              |S|·|E| × 4 bytes
kleene_witnesses:  ceil(log2(|S|)) × |S|² bytes
D_global:          (|E|+|S|)² × 4 bytes     (after un-condensation)
```

### 14.3 Total Memory

```
Transfer matrices + Schur data: O(N² log N) bytes
Kleene witnesses: O(N² log² N) bytes
D_global tables: O(N² log N) bytes

Practical:
  N=64:    < 2 MB
  N=128:   ~10 MB
  N=256:   ~50 MB
  N=1024:  ~1 GB
```

---

## 15. Complete Pipeline

```
function path_circuit(grid_costs, N):
    grid_costs, N = pad_grid(grid_costs, N)

    # Phase 1: Base blocks
    base_blocks = (N/4) × (N/4) array
    for br = 0 to N/4 - 1:
        for bc = 0 to N/4 - 1:
            base_blocks[br][bc] = compute_base_block(br, bc, grid_costs)

    # Phase 2: Hierarchical composition
    root, all_nodes = build_hierarchy(base_blocks, N, grid_costs)

    # Phase 3a: Un-condensation
    L = log2(N / 4)
    un_condense_all(root, L)

    # Build oracle index
    oracle = build_oracle(root, base_blocks)

    return oracle

# Usage:
oracle = path_circuit(grid_costs, N)
d = query_distance(oracle, (r1,c1), (r2,c2))
path = reconstruct_path(oracle, (r1,c1), (r2,c2))
```

---

## 16. Test Cases

| # | Test | Grid | What It Catches |
|---|------|------|-----------------|
| 1 | Empty grid | 4×4, 8×8, 16×16, 32×32, 64×64 | Basic correctness, 2×Manhattan |
| 2 | Single obstacle | 8×8, obstacle at (3,3) | Detour computation |
| 3 | H-wall with gap | 16×16, wall row 8 cols 0–14 | Long detour through hierarchy |
| 4 | V-wall with gap | 16×16, wall col 8 rows 0–14 | Vertical merge correctness |
| 5 | DFS maze | 16×16, 32×32 | Complex topology |
| 6 | Random 25% obstacles | 8–64, 100 random grids | Statistical correctness |
| 7 | Random 50% obstacles | 8–32 | Heavy disconnection |
| 8 | Narrow corridor 4×32 | 4×32 | Minimal height blocks |
| 9 | Fully disconnected | 32×32, complete wall | SENTINEL for cross-wall |
| 10 | Non-uniform [1,100] | 16×16, 32×32 | Weighted paths ≠ BFS |
| 11 | Spiral maze | 64×64 | Max-length paths |
| 12 | Path exits LCA | 16×16 with internal wall | Path leaves containing block |
| 13 | Corner cell routing | 16×16 | Paths through merge corners |
| 14 | One-cell corridors | 8×8 | Extreme sparsity |
| 15 | Zero-cost cells | 8×8 | Edge weight = 0 |
| 16 | Source = goal | any | Must return 0 |
| 17 | Obstacle source | any | Must return SENTINEL |
| 18 | Single block 4×4 | 4×4 | No hierarchy |
| 19 | Path reconstruction | All above | validate_path on random pairs |

**Procedure per test:** Build oracle. Dijkstra from every non-obstacle source. Compare every d(s,g). Zero mismatches. For test 19, reconstruct paths for random pairs and verify cost = d(s,g).

---

## 17. Checklist

```
[ ] tropical_add: clamp to SENTINEL, early-exit on SENTINEL inputs
[ ] Every matrix initialized to SENTINEL, not 0
[ ] Diagonal of W set to 0
[ ] BOUNDARY_INDICES and local_to_boundary are consistent
[ ] Boundary enumeration order identical in build, un-condense, query
[ ] Corner cells excluded from interface set S
[ ] Corner cross-edges included in M_EE
[ ] Kleene closure: ceil(log2(|S|)) squarings, no accumulation
[ ] M_ES stored separately for un-condensation                    [V2]
[ ] All three merge nodes per group stored                         [V2]
[ ] Un-condensation: vertical merge first, then horizontal         [V2]
[ ] propagate_D_global correctly maps child boundary → parent index [V2]
[ ] D_SS uses M_ES (not M_ES_times_MSS_star) in un-condensation   [V2]
[ ] Base block D[i][i] = 0 for all i including obstacles
[ ] Empty 4×4 grid: T[a][b] = 2 × Manhattan(β_a, β_b)
[ ] 8×8 numerical trace matches (§12)
[ ] All test cases pass with zero mismatches
[ ] Symmetry: T[i][j] == T[j][i] at every level
[ ] Zero diagonal: T[i][i] == 0 at every level
[ ] Path reconstruction: validate_path passes on random samples
```
