# Path Circuits: Composable Algebraic Pathfinding on Grid Graphs

---

## 1. The Core Idea

### 1.1 The Problem with Search-Based Pathfinding

Standard shortest path algorithms (Dijkstra, A*, BFS) are iterative search procedures. They maintain a frontier, expand nodes, and repeat until the goal is reached. This has two consequences:

1. **Control flow depends on data.** Which node is expanded next depends on the edge weights. Different obstacle patterns produce different execution traces. This makes the algorithms difficult to parallelize at the hardware level — it is not possible to pipeline or vectorize a computation whose structure changes with the input.

2. **Work is not reusable.** The computation performed for one source-goal pair shares almost nothing with the computation for another pair. Computing all-pairs shortest paths requires repeating the search from every source.

### 1.2 The Alternative: Pathfinding as Algebraic Composition

An alternative computational model replaces search with **algebraic composition of local summaries** — small regions are summarized independently, then combined into global summaries of larger regions using purely algebraic operations.

The central idea has three parts:

**Part 1 — Local summarization.** A small block of the grid (say 4×4 cells) can be completely characterized for pathfinding purposes by its **transfer matrix**: a table of shortest path distances between every pair of cells on its boundary. Once computed, this summary contains everything needed for routing paths through the block, without knowledge of the block's internal structure.

**Part 2 — Algebraic composition.** When two adjacent blocks are combined, their transfer matrices can be composed into a single transfer matrix for the combined region. This composition is a fixed algebraic operation (tropical matrix multiplication and Schur complement) that takes two small matrices and produces one larger matrix. No search, no iteration.

**Part 3 — Hierarchical assembly.** Organizing blocks into a tree (pairs of blocks compose into super-blocks, pairs of super-blocks into mega-blocks, etc.) allows the entire grid to be summarized in O(log N) levels of composition. Each level is fully parallel — all compositions at the same level are independent.

The result is a **feed-forward computational pipeline** with fixed structure. The same sequence of algebraic operations executes regardless of which cells are obstacles, what the costs are, or where the start and goal are. The circuit topology depends only on the grid dimensions.

### 1.3 The Physical Analogy

Think of each 4×4 block as a physical chip with pins on its edges. The chip's internal circuitry (the obstacle/cost pattern) determines how signals (paths) propagate from pin to pin. The transfer matrix is the chip's datasheet — it describes the propagation characteristics without revealing the internal layout.

To build a larger system, chips snap together, matching pins along shared edges. The combined system's behavior is determined by composing the individual datasheets. No chip is ever opened to inspect its internals — only the interface specifications matter.

This composition is:
- **Modular.** Each block is self-contained. Changing one block's internals requires only that block's summary to be recomputed.
- **Scale-free.** A super-block composed from four blocks has the same kind of interface (boundary cells with a transfer matrix) as a base block. The composition operation is the same at every scale.
- **Parallel.** All blocks at the same level are independent and can be computed simultaneously.
- **Feed-forward.** Information flows bottom-up through the tree. No iteration, no feedback, no convergence test.

---

## 2. Mathematical Framework

### 2.1 The Tropical Semiring

All computations take place in the **tropical semiring** T = (ℝ≥0 ∪ {∞}, ⊕, ⊗) where:

$$a \oplus b = \min(a, b) \qquad a \otimes b = a + b$$

- Additive identity: ∞ (since min(a, ∞) = a for all a)
- Multiplicative identity: 0 (since a + 0 = a for all a)

This is a semiring: both operations are associative, ⊗ distributes over ⊕, and the identities satisfy the semiring axioms.

**Why this semiring encodes shortest paths:** The shortest path distance from u to v is:

$$d(u,v) = \min_{\text{paths } \pi: u \to v} \sum_{e \in \pi} w(e) = \bigoplus_{\pi} \bigotimes_{e \in \pi} w(e)$$

Shortest path computation is literally summation and multiplication in the tropical semiring, with "sum" being min over paths and "product" being addition of edge weights along a path.

**Tropical matrix multiplication:** For matrices A (p×q) and B (q×r) over T:

$$(A \otimes B)[i][j] = \bigoplus_{k=1}^{q} (A[i][k] \otimes B[k][j]) = \min_{k} (A[i][k] + B[k][j])$$

This is standard matrix multiplication with (×, +) replaced by (+, min). It computes the shortest two-hop path distances: (A ⊗ B)[i][j] is the minimum over all intermediate vertices k of the cost of going i → k → j.

### 2.2 The Grid Graph

An N×N grid G has N² cells, each with traversal cost c(v) ∈ ℤ≥0 ∪ {∞}. Obstacles have cost ∞. Cells are 4-connected (up, down, left, right neighbors). The edge weight between adjacent cells i and j is:

$$w(i,j) = c(i) + c(j)$$

A path [s, v₁, v₂, ..., vₖ, g] has total cost c(s) + 2c(v₁) + 2c(v₂) + ... + 2c(vₖ) + c(g). Interior cells contribute twice (once per edge); endpoints contribute once.

### 2.3 Transfer Matrix: The Block Summary

**Definition.** For a block B of the grid with boundary cell set ∂B, the **transfer matrix** T_B is the |∂B| × |∂B| matrix over T defined by:

$$T_B[i][j] = d_B(i,j) \quad \text{for } i,j \in \partial B$$

where d_B(i,j) is the shortest path distance from i to j using only cells within B.

**Properties:**
- T_B is symmetric (the grid is undirected).
- T_B[i][i] = 0 for all i (distance from a cell to itself is zero).
- T_B satisfies the triangle inequality: T_B[i][j] ≤ T_B[i][k] + T_B[k][j].
- T_B[i][j] = ∞ if and only if i and j are not connected within B.

**What the transfer matrix captures:** T_B is a **sufficient statistic** for the block's pathfinding behavior. For any graph containing B as a subgraph, the shortest path distance between any two external vertices (outside B) is unchanged if B is replaced by the complete weighted graph on ∂B with weights T_B. The block's interior can be discarded without affecting any external shortest path.

**Theorem (Sufficiency).** Let G be a graph containing block B with boundary ∂B. Let G' be G with B's interior removed and replaced by edges {(i,j) : i,j ∈ ∂B} with weight T_B[i][j]. Then for all u, v ∈ V(G) \ (V(B) \ ∂B):

$$d_G(u,v) = d_{G'}(u,v)$$

**Proof.** Any shortest path in G that enters B at boundary cell i and exits at boundary cell j uses an internal subpath of cost ≥ T_B[i][j] (by definition of shortest path within B), with equality achieved by some internal path. In G', the direct edge (i,j) has weight T_B[i][j]. Paths not entering B are unchanged. ∎

**What the transfer matrix does not capture:** The specific cells on the shortest internal path (only the cost is preserved, not the route), the second-shortest path, the number of shortest paths, or any property of the block's interior not reflected in boundary-to-boundary distances.

---

## 3. Base Block Computation

### 3.1 Setup

A 4×4 block has 16 cells labeled 0–15:

```
 0  1  2  3       B  B  B  B
 4  5  6  7       B  ι  ι  B
 8  9 10 11       B  ι  ι  B
12 13 14 15       B  B  B  B
```

Boundary ∂B = {0,1,2,3,4,7,8,11,12,13,14,15} — 12 cells on the perimeter.
Interior = {5,6,9,10} — 4 cells.

### 3.2 Weight Matrix

The 16×16 matrix W over T is constructed as:

$$W[i][j] = \begin{cases} 0 & i = j \\ c(i) + c(j) & i,j \text{ are 4-adjacent and both non-obstacle} \\ \infty & \text{otherwise} \end{cases}$$

W is the weighted adjacency matrix of the block, with zero-cost self-loops. The self-loops are critical — they allow "padding" short paths to longer walks without increasing cost.

### 3.3 All-Pairs Shortest Paths via Tropical Matrix Powering

**Claim:** D = W^{⊗16} (the 16th tropical power of W) gives exact all-pairs shortest path distances within the block.

**Computation via repeated squaring:**

$$S_1 = W \otimes W \quad S_2 = S_1 \otimes S_1 \quad S_3 = S_2 \otimes S_2 \quad D = S_3 \otimes S_3$$

Four tropical matrix multiplications of 16×16 matrices.

**Why W^{⊗16} suffices:**

(W^{⊗k})[i][j] = the minimum cost of any walk from i to j using exactly k steps, where a "step" is either traversing an edge or staying at the current cell (self-loop, cost 0).

Since W[i][i] = 0, any walk of length ℓ < k can be padded to length k by inserting k − ℓ self-loops at no extra cost. Therefore:

$$W^{\otimes k}[i][j] = \min_{\ell \leq k} (\text{shortest walk of length } \ell) = \min_{\ell \leq k} W^{(\ell)}[i][j]$$

where $W^{(\ell)}$ counts walks of exactly ℓ steps without self-loops.

The longest simple path on 16 vertices has at most 15 edges. Therefore W^{⊗15} captures all shortest paths. Since W^{⊗k} is monotonically non-increasing (more steps can only help or be neutral), W^{⊗16} = W^{⊗15}. And 16 = 2⁴ is the natural output of 4 repeated squarings.

**This is algebraically equivalent to Floyd-Warshall.** The distinction is the computational model: Floyd-Warshall is typically implemented with three nested loops and data-dependent early termination; the tropical matrix powering is a fixed sequence of matrix multiplications — same operations in the same order regardless of the data.

### 3.4 Transfer Matrix Extraction

The 12×12 transfer matrix is the submatrix of D indexed by the boundary cells:

$$T_B[a][b] = D[\beta_a][\beta_b] \quad \text{for } a,b \in \{0,\ldots,11\}$$

where β is the ordered list of boundary cell indices.

The full 16×16 matrix D is also retained — it is needed later for injecting start/goal cells that lie in the block's interior.

### 3.5 Operation Count

Each tropical matrix multiplication of 16×16 matrices: 16³ = 4,096 additions, 16² × 15 = 3,840 comparisons. Four multiplications: ~32,000 operations total per base block. This is fixed — the same for every obstacle pattern.

### 3.6 Witness Matrices

To reconstruct actual paths (not just distances), **midpoint witnesses** are recorded during each squaring step:

$$W_t[i][j] = \arg\min_k \left( S_{t-1}[i][k] + S_{t-1}[k][j] \right)$$

W_t[i][j] stores the midpoint of the optimal walk at squaring step t. Four witness matrices (one per squaring step), each 16×16. Path reconstruction proceeds by looking up the midpoint from W₄, splitting the path in two, recursing using W₃ for each half, and continuing through W₂ and W₁. This is recursive bisection — O(log 16) = 4 recursion levels per path.

---

## 4. Hierarchical Composition

### 4.1 The Composition Problem

Given two adjacent blocks A and B with transfer matrices T_A and T_B, the goal is to compute the transfer matrix T_C of the combined block C = A ∪ B.

### 4.2 Boundary Partition

For a horizontal merge (A on the left, B on the right):

A's right column and B's left column share a physical boundary. The cells on this shared boundary fall into two categories:

- **Corner cells** (top and bottom of the shared column): These sit on the perimeter of the combined block C and must remain as boundary cells of C.
- **Interior interface cells**: All other shared-column cells. These are interior to C and will be eliminated.

Define:
- I_A = right column of A, excluding corners (cells to be eliminated)
- I_B = left column of B, excluding corners (cells to be eliminated)
- S = I_A ∪ I_B (the full set of cells to eliminate)
- E_A = boundary of A minus I_A (cells that stay external)
- E_B = boundary of B minus I_B (cells that stay external)
- E = E_A ∪ E_B (the boundary of the combined block C)

For a level-(k-1) block of height h = 4·2^{k-1}: |I_A| = |I_B| = h − 2, so |S| = 2(h−2) = O(2^k) and |E| = O(2^k).

### 4.3 Joint Matrix Assembly

The distance relationships in the combined block, before eliminating S, are captured by the joint matrix M over the vertex set E ∪ S:

$$M = \begin{pmatrix} M_{EE} & M_{ES} \\ M_{SE} & M_{SS} \end{pmatrix}$$

Populated as follows:

- **Within A**: distances between E_A cells and between E_A and I_A cells come from T_A.
- **Within B**: distances between E_B cells and between E_B and I_B cells come from T_B.
- **Across the interface**: I_A[r] connects to I_B[r] (same row) with cost c(I_A[r]) + c(I_B[r]). These are the cross-edges in C_AB (a diagonal matrix).
- **Corner cross-edges**: corner cells of A's right column that are grid-adjacent to corner cells of B's left column are connected with their edge weights. These entries go into M_EE.
- **All other cross-block entries**: ∞ (no direct connection).

### 4.4 The Tropical Schur Complement

The transfer matrix of C is obtained by **eliminating** the interface vertices S from M, keeping only the external vertices E. This is the tropical analog of Gaussian elimination / Schur complement:

$$T_C = M_{EE} \oplus \left( M_{ES} \otimes M_{SS}^* \otimes M_{SE} \right)$$

where $M_{SS}^*$ is the **Kleene closure** (tropical matrix closure) of $M_{SS}$:

$$M_{SS}^* = \bigoplus_{k=0}^{\infty} M_{SS}^{\otimes k} = I \oplus M_{SS} \oplus M_{SS}^{\otimes 2} \oplus \cdots$$

Since $M_{SS}$ has zeros on its diagonal and non-negative off-diagonal entries, this series converges — specifically, $M_{SS}^{\otimes k}$ stabilizes for $k \geq |S|$ (the longest simple path among |S| vertices has |S|−1 edges, and self-loops pad shorter paths for free).

**Computation:** $M_{SS}^*$ is computed via repeated squaring of $M_{SS}$: ⌈log₂|S|⌉ tropical matrix multiplications of |S|×|S| matrices.

### 4.5 Interpretation

The formula $T_C = M_{EE} \oplus (M_{ES} \otimes M_{SS}^* \otimes M_{SE})$ says:

The shortest path between two external cells u, v ∈ E in the combined block is the minimum of:

1. **$M_{EE}[u][v]$**: The best path that doesn't use any interface vertex in S. This covers paths staying entirely within A, entirely within B, or crossing via corner cells.

2. **$(M_{ES} \otimes M_{SS}^* \otimes M_{SE})[u][v]$**: The best path that enters the interface (S), possibly bounces back and forth through the interface (captured by $M_{SS}^*$), and exits. This covers all paths that use any interface vertex, including zigzag paths (A → B → A → B → ...).

The Kleene closure $M_{SS}^*$ is what handles zigzag paths — it considers all possible numbers of interface crossings.

### 4.6 Correctness of Composition

**Theorem.** T_C as computed above correctly encodes all shortest path distances between external boundary cells of C = A ∪ B.

**Proof.**

*Completeness:* Any path from u ∈ E to v ∈ E within C either (a) avoids all vertices in S, in which case its cost is captured by M_EE[u][v]; or (b) uses some vertices in S, in which case it can be decomposed as: u → (some vertex in S) → (sequence of S vertices) → (some vertex in S) → v. The cost is M_ES[u][s₁] + (path among S vertices from s₁ to s₂) + M_SE[s₂][v], minimized over all s₁, s₂ and all S-internal paths. The Kleene closure M_SS*[s₁][s₂] captures the optimal S-internal path. The tropical matrix products minimize over s₁ and s₂. So case (b) is captured by (M_ES ⊗ M_SS* ⊗ M_SE)[u][v]. The ⊕ takes the better of cases (a) and (b).

*Soundness:* Every term in the formula corresponds to a valid path within C (edge weights come from T_A, T_B, and cross-edges, all of which represent real paths within their blocks). No spurious paths are introduced.

*Induction:* By induction on hierarchy level, applying this composition at every level produces correct transfer matrices at every level. ∎

### 4.7 The Hierarchy

The grid is organized as a balanced quadtree:

```
Level 0:  (N/4)² base blocks, each 4×4,           transfer matrices 12×12
Level 1:  (N/8)² blocks, each 8×8,                 transfer matrices 28×28
Level 2:  (N/16)² blocks, each 16×16,              transfer matrices 60×60
  ⋮
Level L:  1 block covering the full N×N grid        transfer matrix O(N)×O(N)
```

where L = log₂(N/4).

At each level, four blocks are composed into one via three binary merges in the pattern (Horizontal, Horizontal, Vertical):

```
[A] [B]     →   [AB]        →   [ABCD]
[C] [D]     →   [CD]
```

The two horizontal merges are independent and execute in parallel. The vertical merge depends on both and executes after.

### 4.8 Complexity

**Work per level k:** O(N²/4^k) compositions, each dominated by the Kleene closure on an O(2^k) × O(2^k) matrix at cost O(8^k). Per-level total: O(N² · 2^k).

**Total work:** $\sum_{k=1}^{L} O(N^2 \cdot 2^k) = O(N^2 \cdot 2^L) = O(N^3)$

The sum is a geometric series dominated by the top level. The single most expensive operation in the entire computation is the top-level Kleene closure on an O(N) × O(N) interface matrix, costing O(N³).

**Parallel depth per level k:** O(k²) — from O(k) squaring steps for the Kleene closure, each tropical matrix multiply having O(k) depth via parallel reduction.

**Total parallel depth:** $\sum_{k=1}^{L} O(k^2) = O(L^3) = O(\log^3 N)$

**Memory:** Transfer matrices across all levels: O(N² log N). Witness tables: O(N² log² N).

---

## 5. Querying the Oracle

### 5.1 What the Hierarchy Produces

After bottom-up composition (O(N³) work, O(log³ N) depth), every node in the hierarchy tree stores its transfer matrix, Kleene closure, and auxiliary matrices. This is the **hierarchical oracle** — a data structure encoding all shortest path distances implicitly.

### 5.2 Un-Condensation: Recovering Eliminated Distances

The bottom-up phase eliminates interface vertices at each level. To answer queries efficiently, a **top-down pass** reverses these eliminations, recovering the distances involving eliminated vertices.

At each level k (processed from root down to leaves), for each block with external boundary E and eliminated interface S:

**Given:** D_EE_global — the global all-pairs distances among E vertices (available from higher levels; at the root, this is the root's transfer matrix itself).

**Recover:**

$$D_{ES} = D_{EE} \otimes (M_{ES} \otimes M_{SS}^*)$$

$$D_{SS} = M_{SS}^* \oplus (M_{SE} \otimes D_{EE} \otimes M_{ES})$$

The first equation computes distances from external to interface vertices. The second computes distances among interface vertices, accounting for both internal paths (M_SS*) and paths that exit the block and return (M_SE ⊗ D_EE ⊗ M_ES).

**Why one pass suffices (no iteration needed):** D_EE_global was computed from the Schur-complement-reduced transfer matrix T_C, which already encodes S's effect on E-to-E distances. Therefore D_EE_global[e₁][e₂] correctly accounts for paths between e₁ and e₂ that route through S. The "external excursion" term captures all multi-crossing paths without iteration.

**Proof:** For s₁, s₂ ∈ S, any shortest path decomposes into alternating internal segments (through S, cost captured by M_SS*) and external excursions (exit to E via M_SE, traverse external graph, re-enter via M_ES). An external excursion from e₁ to e₂ might itself pass through S — but this is already captured in D_EE_global because T_C folded S into E via the Schur complement during the bottom-up phase. D_EE_global was built using T_C at higher levels. Therefore the single-excursion formula is exact. ∎

**Work:** O(N³) total (same geometric series as the bottom-up phase). **Depth:** O(log² N).

### 5.3 Distance Queries

**After un-condensation**, the oracle answers any distance query d(s,g) in O(log N):

1. If s and g are in the same base block: read d(s,g) directly from the stored 16×16 matrix. O(1).

2. Otherwise: s is in base block B_s with boundary ∂B_s, g is in base block B_g with boundary ∂B_g. Compute:

$$d(s,g) = \min_{b_s \in \partial B_s, \, b_g \in \partial B_g} \left( D_{B_s}[s][b_s] + d_{\text{global}}(b_s, b_g) + D_{B_g}[b_g][g] \right)$$

where D_{B_s}[s][b_s] is from the base block's stored matrix, and d_global(b_s, b_g) is looked up from the un-condensed oracle at the lowest hierarchy level where both b_s and b_g appear in the same block's stored distance table.

The minimization is over 12 × 12 = 144 pairs. Each d_global lookup traverses O(log N) hierarchy levels. Total: O(log N) per query.

Each block's D_global is stored as a dense matrix, so once the correct level is found, the distance read is O(1).

### 5.4 Path Reconstruction

Given the optimal distance and the hierarchy of witness tables, the actual path is reconstructed top-down:

1. The query identifies which boundary cells and which hierarchy level carry the optimal path.
2. At each level, midpoint witnesses identify the intermediate vertex via recursive bisection.
3. Recursion into child blocks handles each sub-segment.
4. At base blocks, the 4 witness matrices reconstruct cell-by-cell paths.

**Total:** O(path_length × log N).

---

## 6. Worked Example: 8×8 Grid

### 6.1 Setup

An 8×8 grid with uniform cost c(v) = 1, and one obstacle at position (2,5):

```
1 1 1 1 | 1 1 1 1
1 1 1 1 | 1 1 1 1
1 1 1 1 | 1 ∞ 1 1
1 1 1 1 | 1 1 1 1
--------+--------
1 1 1 1 | 1 1 1 1
1 1 1 1 | 1 1 1 1
1 1 1 1 | 1 1 1 1
1 1 1 1 | 1 1 1 1
```

Four base blocks: A (top-left), B (top-right, contains obstacle), C (bottom-left), D (bottom-right).

### 6.2 Phase 1: Base Block Transfer Matrices

**Block A (uniform, no obstacles):** All edge weights = 1 + 1 = 2. The 16×16 matrix W_A has W[i][j] = 2 for adjacent pairs, 0 on diagonal, ∞ elsewhere. After 4 squarings, D_A encodes Manhattan distances scaled by 2. Transfer matrix T_A (12×12): T_A[i][j] = 2 × Manhattan_distance(i,j).

Example: T_A[cell 0][cell 3] = 2 × 3 = 6 (three hops across the top row).

**Block B (one obstacle):** Cell 9 (row 2, col 1 within B) has cost ∞. All edges to cell 9 are ∞. Paths that would have crossed cell 9 must detour. For instance, the shortest path from cell 8 to cell 10 within B can no longer go 8→9→10 (cost 4). It must detour: 8→4→5→6→10 or 8→12→13→14→10, costing 8. T_B differs from T_A at entries affected by this detour.

**Blocks C, D:** Identical to A (uniform, no obstacles).

### 6.3 Phase 2: Composition

**Merge 1 (A ∥ B):** A's right column = {3, 7, 11, 15}. Corners: cells 3 (top) and 15 (bottom) remain external. Interface: I_A = {7, 11}, I_B = {4, 8} (B's left column minus corners). |S| = 4.

The joint matrix M is 24×24 (10 + 2 + 2 + 10). M_SS is 4×4:
- I_A internal distances from T_A
- I_B internal distances from T_B (affected by the obstacle)
- Cross-edges: c(A:7)+c(B:4) = 2, c(A:11)+c(B:8) = 2

Kleene closure: 2 squarings of a 4×4 matrix (~224 operations).

Schur complement produces T_AB (20×20) — the transfer matrix for the combined 8×4 block.

**Merge 2 (C ∥ D):** Identical to Merge 1 but with no obstacles. T_CD is 20×20 with uniform-grid distances.

**Merge 3 (AB ⫽ CD):** Vertical merge. Shared horizontal boundary of width 8. |S_vert| = 12. Kleene closure: 4 squarings of a 12×12 matrix (~14,000 operations). Produces T_ABCD (28×28) — the complete 8×8 grid's boundary transfer matrix. The obstacle in B affects distances that route through the top-right quadrant.

### 6.4 Query

Query: s = (0,0), g = (2,6).

s is in Block A (local cell 0). g is in Block B (local cell 10, just right of the obstacle).

From base blocks: v_s = D_A[0][β_i] for all 12 boundary cells of A. v_g = D_B[β_j][10] for all 12 boundary cells of B.

The oracle finds the optimal path: from s through A to the AB interface, across into B, and around the obstacle to reach g. The obstacle forces a detour, so d(s,g) > 2 × Manhattan((0,0),(2,6)) = 16.

---

## 7. Complexity Summary

| Metric | Value |
|--------|-------|
| Oracle build work | O(N³) |
| Oracle build parallel depth | O(log³ N) |
| Un-condensation work | O(N³) |
| Un-condensation depth | O(log² N) |
| Per-query time (after un-condensation) | O(log N) |
| Path reconstruction | O(path_length × log N) |
| Oracle memory (distances) | O(N² log N) |
| Oracle memory (distances + witnesses) | O(N² log² N) |

**Comparison for all-pairs shortest paths on N×N grids:**

| Method | Build Work | Per-Query | Parallel Depth |
|--------|-----------|-----------|---------------|
| Floyd-Warshall | O(N⁶) | O(1) | O(N²) |
| Dijkstra from all sources | O(N⁴) | O(1) | Sequential |
| **Path Circuit** | **O(N³)** | **O(log N)** | **O(log³ N)** |

The Path Circuit builds a compact implicit oracle (O(N² log N) space) rather than the explicit N⁴-entry distance matrix. It produces the same distances — verifiable via exact integer arithmetic — but trades O(1) query time for O(log N) query time, in exchange for an exponentially smaller build time and exponentially better parallelism.

---

## 8. Key Theoretical Properties

### 8.1 Data Independence

The circuit topology — which operations happen in which order on which data positions — depends only on N, not on the cell costs. Every obstacle pattern, every cost configuration, executes the same computation graph. This means:

- No branch mispredictions. No warp divergence on GPU.
- The circuit can be compiled to a static dataflow graph or hardware pipeline for a given N.
- Worst-case performance equals average-case performance.

### 8.2 Integer Exactness

Using integer costs and integer arithmetic with sentinel representation of ∞ (SENTINEL = 2²⁹, with per-addition clamping), the oracle produces distances that are bitwise identical to Dijkstra. No floating-point rounding, no accumulation error, no platform-dependent results.

### 8.3 Algebraic Structure

The transfer matrices form the morphisms of a category:

- **Objects:** Finite ordered sets (boundary cell sets).
- **Morphisms:** Tropical matrices. T_B: ∂B_left → ∂B_right.
- **Composition:** Tropical matrix multiplication (with Schur complement for elimination).
- **Identity:** Tropical identity matrix (0 on diagonal, ∞ elsewhere).

Composition is associative (inherited from tropical matrix multiplication). The monoidal product (placing blocks side-by-side with no connection) is the block diagonal construction. This means:

- Composition order does not matter (any sequence producing the same block yields the same transfer matrix).
- Blocks of any compatible shape can be composed (4×8, 16×4, arbitrary rectangles).
- The framework extends naturally to non-square grids, L-shaped regions, or any region decomposable into rectangular blocks.

### 8.4 Modularity

If one cell changes, only its base block's transfer matrix requires recomputation (O(1) work), followed by the O(log N) ancestor compositions along the path from that block to the root. The rest of the hierarchy is unchanged. This gives incremental update cost equal to the cost of the top-level recomposition: O(N³) worst case (dominated by the top-level Kleene closure on the O(N) interface), but with practical savings when only lower levels are affected.

---

## 9. Extensions and Directions

### 9.1 Differentiable Path Circuits

**The substitution:** Replace the hard minimum with a smooth approximation:

$$\min(a, b) \;\longrightarrow\; \text{softmin}_\tau(a, b) = -\frac{1}{\tau} \log\!\left(e^{-\tau a} + e^{-\tau b}\right)$$

where τ > 0 is a temperature parameter. As τ → ∞, softmin → min. For finite τ, softmin is smooth and infinitely differentiable.

The + operation is already smooth. Therefore, replacing every min with softmin makes the entire pipeline — base block matrix powering, Kleene closure, Schur complement, un-condensation — a **smooth differentiable function** from cell costs to path distances:

$$f_\tau : \mathbb{R}^{N^2} \to \mathbb{R}^{|\partial G|^2}, \quad \mathbf{c} \mapsto \tilde{T}_\text{root}$$

The function $f_\tau$ maps a cost vector (one cost per cell) to the soft transfer matrix of the full grid. For large τ, this approximates the exact shortest path distances. Crucially, the Jacobian ∂f_τ/∂c exists everywhere and can be computed by backpropagation through the same circuit.

**What this enables:**

*Inverse pathfinding.* Given observed shortest paths (demonstrations), the cost map that makes those paths optimal can be learned. A loss function measuring discrepancy between observed paths and optimal paths under the current cost map is defined, and backpropagation through the oracle updates the costs. This is the spatial-planning analog of inverse reinforcement learning.

*End-to-end perception-to-planning.* The differentiable oracle can serve as a layer in a neural network. A perception module (CNN) processes raw sensor data and outputs a cost grid. The oracle layer computes soft shortest path distances. The loss is defined on path quality or downstream task performance. Gradients flow from the loss through the oracle back to the perception module, enabling end-to-end training.

*Gradient-based cost map optimization.* The cost map can be directly optimized for desirable global properties: minimizing average path length across all pairs, maximizing connectivity, balancing traffic flow. The oracle makes such objective functions differentiable with respect to every cell cost.

**Open theoretical questions:**

- *Bias accumulation:* Softmin is an approximation. At each hierarchy level, the approximation introduces a small bias (slightly overweighting suboptimal paths). Whether this bias accumulates, stabilizes, or cancels through the hierarchy is unknown. Characterizing the approximation error ‖f_τ(c) − f_∞(c)‖ as a function of τ and the hierarchy depth is an open problem.

- *Gradient survival:* Whether the gradient signal ∂f_τ/∂c vanishes or explodes when backpropagated through O(log³ N) levels of softmin matrix operations is an open question. The answer likely depends on τ and the cost structure.

- *Optimal temperature schedule:* Whether τ should be the same at every hierarchy level or should vary (e.g., τ_k = τ₀ · 2^k) to maintain consistent approximation quality as the matrix dimensions grow is unexplored.

### 9.2 Hardware Realization

The entire pipeline is a fixed dataflow graph for any given N. This allows direct mapping to hardware:

- **FPGA:** Each hierarchy level becomes a pipeline stage. Each composition unit is a tropical matrix multiplier (comparator + adder arrays). Latency: O(log³ N) clock cycles from grid input to oracle output. Throughput: one oracle per clock cycle (fully pipelined).

- **ASIC:** A dedicated "path chip" that takes an N×N cost grid as input and produces the oracle in fixed time. For N = 64: ~1000 base block units, ~300 composition units, operating at GHz clock rates.

- **GPU:** Each hierarchy level maps to a single kernel launch. All blocks at the same level execute as parallel thread blocks. The fixed topology eliminates warp divergence entirely.

The tropical operations (min, +) are simpler than floating-point multiply-accumulate. A tropical matrix multiplier uses only comparators and adders — simpler, smaller, and lower power than FMA units.

### 9.3 Non-Grid Domains

The algebraic framework — summarize blocks by boundary transfer matrices, compose via tropical Schur complement — applies to any graph admitting a hierarchical separator decomposition:

- **Planar graphs:** By the planar separator theorem (Lipton-Tarjan 1979), any n-vertex planar graph has a separator of size O(√n). This gives a hierarchical decomposition analogous to the grid quadtree, with O(√n)-sized interfaces at each level.

- **Road networks:** Real road networks have small separators due to geographic locality. Hierarchical decomposition is already used in practice (Contraction Hierarchies). The tropical circuit formulation would provide a fixed-topology variant.

- **3D grids:** A k×k×k grid has O(k²) boundary cells per block. The transfer matrices are larger, but the composition algebra is identical. Total work: O(k⁹) = O(N³) where N = k³ is the total cell count.

- **Surface meshes:** Triangle meshes on 2D surfaces are planar graphs. Geodesic distance computation via the tropical circuit would give an all-pairs geodesic oracle.

### 9.4 Tropical Geometry and Compression

The space of realizable transfer matrices for a 4×4 block has at most 16 degrees of freedom (one per cell cost) in a 66-dimensional space (the 12×12 symmetric matrix minus diagonal). This means the transfer matrix entries satisfy tropical polynomial relations — they are not independent.

If these relations can be made explicit, transfer matrices could be compressed: instead of storing all 66 entries, a smaller set of parameters would suffice. At higher hierarchy levels, similar (though harder to characterize) redundancies exist.

The **tropical rank** of a transfer matrix — the minimum r such that T ≈ U ⊗ V with U (m×r) and V (r×m) — determines the compression potential. If grid pathfinding transfer matrices have low tropical rank (bounded by the number of topologically distinct "channels" through the block), then:

- Storage reduces from O(b²) to O(b·r) per block.
- Composition reduces from O(b³) to O(b·r²) per merge.
- Total work could drop below O(N³).

Whether this tropical rank is indeed low for grid pathfinding transfer matrices is an open conjecture.

### 9.5 Cross-Disciplinary Connections

The hierarchical tropical composition framework has precise mathematical correspondences to structures studied across multiple fields:

**Statistical mechanics (renormalization group):** The elimination of internal vertices and retention of boundary interactions is structurally identical to real-space renormalization on a lattice. The tropical semiring is the zero-temperature (β → ∞) limit of the Boltzmann partition function semiring — a connection formalized by Maslov dequantization. RG concepts of fixed points and universality suggest that many different obstacle patterns may produce similar coarse-grained transfer matrices.

**Electrical networks:** The tropical Schur complement parallels Kron reduction in resistor networks, where internal nodes are eliminated to produce effective resistances between boundary nodes. Series composition (cost addition) and parallel composition (minimum of costs) are the tropical analogs of series and parallel resistance.

**Finite element methods:** The hierarchical elimination of internal degrees of freedom is precisely static condensation (substructuring) in FEM, performed over the tropical semiring instead of the real field. Techniques from FEM for optimal decomposition, adaptive partitioning, and error estimation have tropical analogs that are largely unexplored.

**Category theory:** The transfer matrices are morphisms in the category of tropical matrices (Mat(T)), with composition being tropical matrix multiplication. This forms a symmetric monoidal category, guaranteeing that any two valid composition orders yield the same result (coherence), and providing a formal framework for reasoning about the composability of the architecture.

**VLSI timing analysis:** Static timing analysis computes critical paths (longest delays) using the (max, +) semiring — the dual of (min, +). The EDA industry performs exactly this hierarchical block characterization on circuits with billions of gates, providing a proven template for industrial-scale implementation.
