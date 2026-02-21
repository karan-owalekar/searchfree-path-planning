# Implementation Documentation: Search-Free Path Planner

This document details how the search-free algebraic pathfinding algorithm is implemented in code. It covers every module in `pathcircuit/core/`, `pathcircuit/gpu/`, `pathcircuit/grid.py`, and `pathcircuit/warmup.py`, explaining what each piece does, why it was done that way, and how the parts connect.

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
9. [GPU Acceleration](#9-gpu-acceleration)
10. [Pipeline Orchestration](#10-pipeline-orchestration)
11. [Warmup and JIT Caching](#11-warmup-and-jit-caching)
12. [Design Decisions and Trade-offs](#12-design-decisions-and-trade-offs)

---

## 1. Project Layout

The algorithm lives entirely within the `pathcircuit/` package. The core computation is separated from the UI, comparison, and GPU modules:

```
pathcircuit/
├── grid.py              # Grid data structure (shared by all modules)
├── warmup.py            # Eager JIT / GPU kernel compilation
├── core/
│   ├── tropical.py      # Tropical semiring primitives (min-plus matmul, JIT-compiled)
│   ├── base_block.py    # Phase 1: 4×4 base block APSP + transfer matrices
│   ├── hierarchy.py     # Phase 2: hierarchical merge (Schur complement, JIT assembly)
│   ├── uncondense.py    # Phase 3a: top-down distance recovery
│   ├── oracle.py        # Phase 3b: flat boundary table, linked-list retrieval, JIT kernels
│   └── pipeline.py      # Orchestrator: ties all phases together (parallel build)
├── gpu/
│   ├── __init__.py      # CUDA environment setup (preloads NVIDIA libs)
│   ├── boundary_gpu.py  # GPU boundary table build (CUDA shared-memory kernel)
│   ├── oracle_gpu.py    # GPU path reconstruction (successor-field CUDA kernels)
│   └── dijkstra_gpu.py  # GPU Dijkstra via parallel Bellman-Ford
├── compare/
│   ├── runner.py        # Algorithm comparison harness
│   ├── astar.py         # A* baseline (Numba JIT)
│   └── dijkstra.py      # Dijkstra baseline (Numba JIT)
```

The algorithm proceeds in four sequential phases:

1. **Base block computation** — solve all-pairs shortest paths within every 4×4 tile.
2. **Hierarchical composition** — merge tiles bottom-up using the tropical Schur complement.
3. **Un-condensation** — propagate global distances top-down through the hierarchy.
4. **Oracle construction** — build flat boundary distance table, and optionally linked-list structures for near-instant path retrieval.

---

## 2. Dependencies and Why They Were Chosen

Defined in `pyproject.toml`:

| Dependency | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.24 | All matrix operations (tropical matmul, Kleene closure, distance tables). NumPy's vectorised broadcasting is the fallback path; the hot loops are JIT-compiled by Numba. |
| `numba` | >= 0.59 | JIT compilation of all performance-critical kernels. Provides `@njit(cache=True)` for ahead-of-time compilation, `prange` for parallel loops, and `nogil=True` for releasing the GIL in threaded contexts. |
| `pygame-ce` | >= 2.5 | UI only — not used by the algorithm. |
| `cupy-cuda12x` | >= 13.0 | Optional GPU acceleration via `cp.RawKernel` for custom CUDA C kernels. Only loaded when `--gpu` is passed. |

**Why Numba:** The algorithm's inner loops (tropical matmul, hierarchy expansion, boundary table construction) are tight numerical kernels operating on contiguous NumPy arrays. Numba's `@njit` compiles these to native machine code with zero Python overhead. Key features used:
- `cache=True` — persists compiled bytecode to `__pycache__/*.nbi`/`.nbc` files, so subsequent launches skip compilation.
- `nogil=True` — releases Python's GIL, enabling true parallelism with `ThreadPoolExecutor`.
- `prange` — parallel for-loops that Numba distributes across CPU cores.
- Numba-compatible data structures only (NumPy arrays, scalars, tuples) — no Python dicts or lists in hot paths.

**Why CuPy over PyCUDA/raw CUDA:** CuPy's `RawKernel` provides the same low-level control as raw CUDA C (custom kernels, shared memory, atomics) with a much simpler Python API. It also handles NVRTC kernel compilation, memory management, and stream synchronization automatically.

**Why int32:** The SENTINEL value is 2²⁹ = 536,870,912. Two SENTINEL values summed stay below the int32 max of 2³¹ − 1, so overflow is impossible. Intermediate sums during matmul use int64 to prevent overflow before clipping.

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

The grid supports both 4-connected (cardinal directions only) and 8-connected (cardinal + diagonal) movement. This is set at construction time via the `connectivity` parameter (4 or 8). Neighbor enumeration uses precomputed offset lists.

### Grid Padding

The hierarchical algorithm requires the grid dimensions to be a power-of-2 multiple of 4 (so the grid can be recursively halved into 4×4 base blocks). The `to_padded()` method computes the smallest valid dimension ≥ max(rows, cols) and creates a square padded grid. Padding cells are filled with SENTINEL (treated as obstacles), so they never participate in shortest paths.

---

## 4. Tropical Semiring Operations

**File:** `pathcircuit/core/tropical.py`

The entire algorithm is built on the **tropical semiring** (ℝ≥0 ∪ {∞}, min, +), where:
- "Addition" is `min(a, b)`
- "Multiplication" is `a + b`
- The additive identity (zero) is `∞` (SENTINEL)
- The multiplicative identity (one) is `0`

### Matrix Multiplication

```python
def tropical_matmul(A, B):
    # C[i,j] = min_k (A[i,k] + B[k,j])
```

Two implementations exist, selected automatically:

**Numba JIT path** (`_matmul_jit`): A triple-nested loop compiled to native code. Decorated with `@njit(cache=True, nogil=True)` so it releases the GIL and can run in parallel threads. This is the hot path used during oracle construction.

**NumPy fallback path**: Uses broadcasting `a[:, :, np.newaxis] + b[np.newaxis, :, :]` to create a 3D tensor, then takes `np.min` along the middle axis. Used only if Numba is unavailable.

### Witness Recording

`tropical_matmul_with_witness` returns `(C, W)` where `W[i,j] = argmin_k`. The Numba path (`_matmul_witness_jit`) also uses `nogil=True` for parallel execution. Witnesses are used for path reconstruction via recursive bisection within base blocks.

### Min-Plus Matrix-Vector Product

`minplus_mv(D, bi, bd, n)` computes a matrix-vector product used during boundary table construction: `d_new[k] = min_b(bd[b] + D[bi[b], k])`. JIT-compiled with `nogil=True`.

---

## 5. Base Block Computation

**File:** `pathcircuit/core/base_block.py`

This is Phase 1. The padded N×N grid is divided into (N/4)² non-overlapping 4×4 tiles. For each tile:

1. The 16×16 **weight matrix** `W` is constructed via `_build_weight_matrix_jit` — a Numba JIT kernel that takes a flattened costs array, avoiding Python-level grid lookups.
2. APSP is computed via 4 rounds of tropical repeated squaring: `D = W¹⁶`.
3. The 12×12 **transfer matrix** `T` is extracted from the boundary rows/columns of `D`.

### Cell Layout

```
 0  1  2  3       B  B  B  B
 4  5  6  7       B  ι  ι  B
 8  9 10 11       B  ι  ι  B
12 13 14 15       B  B  B  B
```

12 boundary cells (B), 4 interior cells (ι). `BOUNDARY_INDICES = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]`.

### Neighbour Tables

`NEIGHBORS_4` and `NEIGHBORS_8` are precomputed as Numba-compatible flat arrays (`_NEIGH_FLAT_4`, `_NEIGH_FLAT_8`, `_NEIGH_COUNT_4`, `_NEIGH_COUNT_8`) for use in JIT kernels that cannot access Python dicts.

### Parallelization

Base block computation is parallelized in the pipeline via `ThreadPoolExecutor` when the number of blocks exceeds 64. Each block's computation is independent, and the `nogil=True` on tropical matmul/witness kernels ensures true parallel execution without GIL contention.

---

## 6. Hierarchical Composition

**File:** `pathcircuit/core/hierarchy.py`

This is Phase 2. Base blocks are merged bottom-up in a quadtree: four adjacent blocks form a 2×2 group merged via three binary merges (two horizontal, one vertical).

### Joint Matrix Assembly — JIT-Compiled

The merge's central computation is the **joint matrix** `M`, an (|E| + |S|) × (|E| + |S|) matrix. This assembly was a major Python bottleneck because it involves iterating over all cell pairs, checking adjacency, and looking up transfer matrix entries.

The function `_assemble_jit` is a Numba JIT kernel that receives pre-flattened NumPy arrays (boundary cell types, indices, coordinates, costs) and fills the joint matrix entirely in compiled code. The Python wrapper `_assemble_joint_matrix` converts the Python-level lists and dicts into these flat arrays, then dispatches to the JIT kernel.

**Why this matters:** For a 64×64 grid, the hierarchy has ~255 nodes with matrices up to 252×252. The original pure-Python assembly was the dominant cost of hierarchy construction (~1000ms). The JIT version reduces this to ~50ms.

### Tropical Schur Complement

The transfer matrix for the merged block:

```
T_C = M_EE ⊕ (M_ES ⊗ M_SS* ⊗ M_SE)
```

`M_ES_times_MSS_star` is stored on the MergeNode for reuse during un-condensation.

---

## 7. Un-Condensation

**File:** `pathcircuit/core/uncondense.py`

Phase 3a. Top-down recovery of interface distances eliminated by the Schur complement. Starting from the root (whose `D_global = T`), each merge node recovers:

```
D_ES = D_EE ⊗ (M_ES ⊗ M_SS*)
D_SE = D_ES^T
D_SS = M_SS* ⊕ (M_SE ⊗ D_EE ⊗ M_ES)
```

The expanded `D_global` (boundary + interface cells) is propagated to child nodes via `propagate_D_global`, which extracts the sub-matrix corresponding to each child's boundary cells.

---

## 8. Oracle and Query System

**File:** `pathcircuit/core/oracle.py`

This is the largest and most performance-critical module. It contains:
- The flat boundary distance table (replaces hierarchy traversal at query time)
- Linked-list path retrieval structures (atlas + successor table)
- All JIT-compiled query and build kernels

### The Oracle Data Structure

```python
@dataclass
class Oracle:
    base_blocks: list[list[BaseBlock]]
    all_D: np.ndarray                    # [n_blocks, 16, 16] all base block D matrices
    N: int
    grid: Grid
    boundary_dist: np.ndarray            # [B, B] flat boundary-to-boundary distances
    block_bids: np.ndarray               # [blocks_r, blocks_c, 12] boundary IDs per block
    costs_flat: np.ndarray               # [N*N] flattened grid costs for JIT
    atlas: np.ndarray | None             # [blocks_r, blocks_c, 12, 12, 9] path atlas
    successor: np.ndarray | None         # [B, B] boundary successor table
    boundary_coords: np.ndarray | None   # [B, 2] global coords per boundary cell
    boundary_local_bi: np.ndarray | None # [B] local boundary index per cell
```

### Flat Boundary Distance Table

The key architectural insight: after un-condensation, every boundary cell's global distances are scattered across `D_global` tables in the hierarchy. Instead of traversing the hierarchy at query time, all boundary-to-boundary distances are extracted into a single `B×B` flat table during build.

For a 64×64 grid: B = 3072 boundary cells → ~36MB table. This eliminates all hierarchy traversal from queries, reducing distance lookups to O(1) array reads.

#### Build Algorithm — Hierarchy Expansion

For each of the B source boundary cells, distances are propagated through the entire hierarchy tree by iterating over all tree nodes in a specific order:

1. **Tree linearization** (`_linearize_nodes`): All hierarchy nodes are collected in DFS order. Their `D_global` matrices, cell indices, and sizes are flattened into contiguous NumPy arrays (`tree_D_flat`, `tree_cell_flat`, etc.) for JIT access.

2. **Ancestor chain ordering**: For each base block, its ancestor nodes are processed first (nearest to farthest), followed by all remaining nodes. This ensures distances propagate outward from the source through its immediate hierarchy before reaching distant parts of the grid.

3. **JIT expansion kernel** (`_jit_expand_indexed`): For each tree node, finds which of the node's cells already have known distances ("bridge cells"), then computes `d_new[k] = min(d[k], min_b(bridge_d[b] + D_global[b, k]))` for all cells in the node.

4. **Per-block processing** (`_process_block_boundary`): Processes all 12 boundary cells of one block. Uses a pre-allocated scratch buffer (`d_buf`, int64[N*N]) that is reset and reused for each of the 12 boundary cells, avoiding 12 separate allocations per block. Decorated with `@njit(nogil=True)` for parallel execution.

5. **Parallel dispatch**: When there are ≥4 blocks, a `ThreadPoolExecutor` submits all blocks as independent tasks. Since `_process_block_boundary` releases the GIL, threads run truly in parallel.

#### Index Array Optimization

An earlier implementation linearized the full ancestor chain data for each block, copying large arrays. The current approach precomputes `node_to_valid_idx` (mapping node identity to its index in the linearized tree) during `build_oracle`, then constructs small int32 index arrays (`chain_indices`, `remaining_indices`) per block. These index arrays are passed directly to `_jit_expand_indexed`, which uses them to select which nodes to process from the shared linearized tree — eliminating all redundant data copying.

### Distance Query

`query_distance(oracle, s, g)` computes the shortest path distance using 144 flat table lookups:

```python
@njit(cache=True)
def _jit_query_distance(s_local, g_local, D_s, D_g, s_bids, g_bids, boundary_dist, BI):
    best = SENTINEL
    for a in range(12):
        d_sa = D_s[s_local, BI[a]]
        if d_sa >= SENTINEL: continue
        ba = s_bids[a]
        for b in range(12):
            d_bg = D_g[BI[b], g_local]
            if d_bg >= SENTINEL: continue
            bb = g_bids[b]
            total = d_sa + boundary_dist[ba, bb] + d_bg
            if total < best:
                best = total
    return best
```

Three terms: (1) source to source-block boundary via base block D, (2) boundary-to-boundary via flat table, (3) goal-block boundary to goal via base block D. All O(1) array reads. Total: 144 iterations of 3 reads + 2 additions + 1 comparison. Runs in ~1–2μs.

### Path Reconstruction — Two Modes

The oracle supports two reconstruction strategies, selected by the `--optimize` flag:

#### Mode 1: Linked-List Retrieval (`--optimize path`)

Built during oracle construction at the cost of ~60–140ms extra build time. Three structures enable ~2–3μs path retrieval:

**Intra-Block Path Atlas** (`_build_atlas`):
- `int8[blocks_r, blocks_c, 12, 12, 9]` — for every base block, stores the cell-by-cell path between each pair of boundary cells.
- Built via greedy walk using the base block's D matrix (exact distances guarantee optimality).
- ~324KB for a 64×64 grid (fits in L2 cache).

**Boundary Adjacency List** (`_build_adjacency_flat`):
- CSR-format arrays (`adj_neighbors`, `adj_costs`, `adj_offsets`) representing the boundary graph.
- Edges include intra-block transfers (via the block's transfer matrix T) and cross-block grid adjacency.
- ~320KB for a 64×64 grid.

**Boundary Successor Table** (`_build_successor_jit`):
- `int16[B, B]` — `successor[target, current]` gives the next boundary cell on the shortest path from `current` toward `target`.
- Built via `@njit(parallel=True)` with `prange` over target cells (embarrassingly parallel).
- Uses the flat `boundary_dist` table and adjacency list: for each (current, target) pair, finds the neighbour minimizing `edge_cost + boundary_dist[neighbour, target]`.
- ~18MB for a 64×64 grid. Column for one target (6KB) fits in L1 cache.

**Retrieval algorithm** (`_jit_retrieve_linked`):

1. **Gateway selection**: Same 144-iteration loop as distance query, but also records the best source-block and goal-block boundary cells (a\*, b\*).
2. **First mile**: Greedy walk within the source block from `s` to gateway `a*` using the base block D matrix.
3. **Highway traversal**: Follow `successor[b*, current]` from `a*` toward `b*`. At each hop:
   - Cross-block step: the two boundary cells are grid-adjacent, just append.
   - Intra-block transit: read the atlas for the interior cells between the two boundary cells on the same block.
4. **Last mile**: Greedy walk within the goal block from gateway `b*` to `g`.

All operations are precomputed array reads. The JIT kernel runs in ~2–3μs.

#### Mode 2: Inline Greedy Walk (`--optimize oracle`)

Skips building the atlas and successor table, yielding faster oracle build (~60–140ms saved). Path reconstruction falls back to `_jit_reconstruct_inline`:

1. Precompute `d(cell, goal)` for all cells in the goal's base block.
2. Starting at `s`, at each step examine all neighbours.
3. For each neighbour, compute `edge_weight(current, n) + d(n, goal)` where `d(n, goal)` uses the 144-lookup formula through the flat boundary table.
4. Step to the neighbour with the smallest total cost.

This is correct because the flat boundary table provides exact distances. Each step does ~144 lookups × 8 neighbours = ~1152 operations. For a path of length L, total is ~1152L operations, running in ~60–400μs depending on path length.

---

## 9. GPU Acceleration

### GPU Boundary Table Construction

**File:** `pathcircuit/gpu/boundary_gpu.py`

When `--gpu` is active, the boundary table construction — the most expensive oracle build phase (~60% of total time) — runs on the GPU via a custom CUDA kernel.

**Architecture:** One CUDA block per source boundary cell (3072 blocks for 64×64), with 32 cooperative threads per block.

**Key design choices:**

- `d_arr` (the per-source scratch array, 4096 × int32 = 16KB) lives in **shared memory**, providing ~100x faster access than global memory. With 16KB per block and ~100KB shared memory per SM, each SM can host ~6 concurrent blocks.

- **Two-pass node expansion** for correctness under parallelism:
  - *Phase 1 (gather)*: All threads cooperatively scan the node's cells, using `atomicAdd` to build a shared list of cells with known distances.
  - *Phase 2 (scatter)*: All threads cooperatively update target cells using the gathered known-cell list. `__syncthreads()` between phases ensures consistency.

  This is necessary because multiple threads updating `d_arr` in parallel could cause data races. The two-pass approach is analogous to Jacobi iteration (read all, then write all) vs Gauss-Seidel (interleaved reads and writes).

- **Chain-first ordering**: Each block's ancestor chain indices are concatenated into flat arrays with per-block offsets, maintaining the bottom-up ordering that ensures correct distance propagation.

**Performance:** ~2x speedup over CPU boundary table construction (~170ms vs ~360ms total oracle build for 64×64).

### GPU Path Reconstruction

**File:** `pathcircuit/gpu/oracle_gpu.py`

A successor-field architecture for `--optimize oracle` mode:

1. **Layer 1** (`dist_to_goal` kernel): Computes `d(cell, goal)` for every cell on the grid in parallel. Each CUDA block handles one base block (16 threads for 16 cells). Mid-distances are loaded into shared memory for reuse.

2. **Layer 2** (`build_successor` kernel): Builds a successor field where each cell points to the neighbour minimizing `edge_cost + d(neighbour, goal)`. One thread per cell.

3. **Layer 3** (CPU): Follows successor pointers from source to goal. Zero-cost per step (just integer array reads).

**Note:** When `--optimize path` is active (linked-list structures available), the CPU linked-list retrieval (~2μs) is used instead of GPU reconstruction (~3ms), even when `--gpu` is set. The GPU reconstruction is only used in `--optimize oracle` mode where no linked-list exists.

### GPU Dijkstra

**File:** `pathcircuit/gpu/dijkstra_gpu.py`

Parallel Bellman-Ford via CUDA, used for comparison only:
- Relaxation kernel uses `atomicMin` for thread-safe distance updates.
- Iterates until convergence (up to `rows * cols` iterations).
- Path traceback on CPU.

### CUDA Environment Setup

**File:** `pathcircuit/gpu/__init__.py`

`_setup_cuda_env()` preloads pip-installed NVIDIA CUDA libraries (`libnvrtc.so.12`, `libnvrtc-builtins.so.12.9`) via `ctypes.CDLL` with `RTLD_GLOBAL` mode. This is necessary when using CuPy with pip-installed CUDA libraries (as opposed to system-wide CUDA toolkit), because CuPy's `dlopen` calls need the libraries to be globally visible.

---

## 10. Pipeline Orchestration

**File:** `pathcircuit/core/pipeline.py`

`build_pathcircuit_oracle(grid, *, build_linked_list=True, use_gpu=False)` ties all phases together:

```python
def build_pathcircuit_oracle(grid, *, build_linked_list=True, use_gpu=False):
    padded = grid.to_padded()
    N = padded.rows

    # Phase 1: base blocks (parallelized if >= 64 blocks)
    if n_blocks >= 64 and HAS_NUMBA:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # submit compute_base_block for each (br, bc)
    else:
        # sequential computation

    # Phase 2: hierarchical composition
    root, all_nodes = build_hierarchy(base_blocks, N, padded)

    # Phase 3a: un-condensation
    un_condense_all(root, N)

    # Phase 3b: oracle construction
    return build_oracle(root, base_blocks, N, padded,
                        build_linked_list=build_linked_list,
                        use_gpu=use_gpu)
```

The `build_linked_list` flag controls whether the atlas and successor table are built (`--optimize path` vs `--optimize oracle`). The `use_gpu` flag routes boundary table construction to the CUDA kernel.

---

## 11. Warmup and JIT Caching

**File:** `pathcircuit/warmup.py`

Numba's `cache=True` persists compiled bytecode to disk, but the first-ever compilation of each kernel takes 1–3 seconds. The warmup module pre-compiles everything on startup:

**`warmup_jit()`**: Builds two oracle instances on a tiny 8×8 grid — one with `build_linked_list=True` and one with `build_linked_list=False` — for both 4-connected and 8-connected modes. This ensures all JIT kernels for both `--optimize` modes are compiled:
- `_jit_query_distance`, `_jit_reconstruct_inline` (oracle mode)
- `_jit_retrieve_linked`, `_build_successor_jit` (path mode)
- `_process_block_boundary`, `_jit_expand_indexed`
- `_matmul_jit`, `_matmul_witness_jit`, `minplus_mv`
- Comparison baselines: `dijkstra`, `astar`

**`warmup_gpu()`**: Triggers NVRTC compilation of all CUDA `RawKernel` objects by running a tiny GPU oracle build and reconstruction. This avoids a multi-second stall on the first real GPU query.

**Cache invalidation:** When source code changes (e.g., modifying a `@njit` function), Numba's cached `.nbi`/`.nbc` files may become stale. If performance unexpectedly regresses after code changes, clearing the cache is necessary:
```bash
find . -path '*/__pycache__/*' \( -name '*.nbi' -o -name '*.nbc' \) -delete
```

---

## 12. Design Decisions and Trade-offs

### Why 4×4 Base Blocks?

The block size of 4 is the sweet spot:
- **Small enough** that APSP by repeated squaring (4 rounds of 16×16 matmul) is cheap.
- **Large enough** that the ratio of interior to boundary cells (4:12) is meaningful — the Schur complement actually eliminates vertices and compresses information.
- **Powers of 2** make the quadtree structure clean: each base block is 4×4, each level-0 merge produces 8×8, then 16×16, etc.

### Flat Boundary Table vs Hierarchy Traversal at Query Time

An earlier implementation traversed ancestor chains at query time, computing min-plus matrix-vector products through the hierarchy for each query. This was correct but slow (~0.5ms per distance query) because it did meaningful computation at query time.

The flat boundary table eliminates this entirely: all B×B boundary distances are precomputed during build and stored in a single array. Queries become 144 flat array reads (~1–2μs). The trade-off is memory: for 64×64, the table is ~36MB. For 128×128 it would be ~576MB, which would require sparse or hierarchical approaches.

### Linked-List Retrieval vs Inline Greedy Walk

Both are correct (they use exact distances). The trade-off is oracle build time vs query time:

| | Build Cost | Path Retrieval |
|---|---|---|
| Linked-list (`--optimize path`) | +60–140ms (atlas + successor) | ~2–3μs |
| Inline greedy (`--optimize oracle`) | — | ~60–400μs |

The linked-list approach stores precomputed "next hop" answers, reducing path retrieval to pointer chasing. The inline approach computes these answers on the fly. For interactive use (many queries on the same map), linked-list is superior. For frequently changing maps (obstacle drawing), the faster oracle build of the inline approach may be preferred.

### CPU Linked-List over GPU Reconstruction

When `--optimize path` is active, the CPU linked-list retrieval (~2μs) is used even with `--gpu` enabled, because:
- The GPU successor-field reconstruction requires computing `d(cell, goal)` for ALL cells on the grid, then building a successor field — ~3ms of GPU work.
- The CPU linked-list only touches cells along the actual path, doing ~50–70 precomputed array reads.
- The linked-list approach is 1000x faster because it does dramatically less work.

### Buffer Reuse in Boundary Table Construction

Each source boundary cell's expansion needs an `int64[N*N]` scratch array initialized to SENTINEL. The original implementation allocated this inside the JIT kernel for each of the 12 boundary cells per block (12 allocations × 32KB = 384KB per block). The current implementation pre-allocates one buffer per block and reuses it across all 12 boundary cells, with an explicit `for i in range(nn): d_buf[i] = SENTINEL` reset loop. This reduces allocation pressure by 12x.

### GPU Shared Memory for Boundary Table

The GPU boundary table kernel places each thread's `d_arr` in shared memory (16KB per CUDA block) rather than global memory. This is critical because the hierarchy expansion accesses `d_arr` with random patterns (indexed by cell coordinates from the tree). Shared memory provides ~100x lower latency than global memory for these random accesses. The trade-off is occupancy: only ~6 blocks per SM fit in shared memory, but with 3072 blocks total and ~128 SMs, the GPU is still well-utilized.

### Why SENTINEL = 2²⁹?

The value must be large enough to represent "infinity" (unreachable) but small enough that `SENTINEL + SENTINEL` doesn't overflow int32 (max 2³¹ − 1). Since 2²⁹ + 2²⁹ = 2³⁰ < 2³¹ − 1, this satisfies both constraints. It also leaves room for legitimate large distances: a 64×64 grid with all costs = 1 has a maximum shortest-path distance of about 2 × 128 = 256, far below SENTINEL.

### nogil=True on Hot Kernels

All Numba kernels called from `ThreadPoolExecutor` threads are decorated with `nogil=True`. Without this, Python's GIL serializes the threads, negating the parallelism. With `nogil=True`, Numba releases the GIL before entering the compiled code, enabling true concurrent execution. This is safe because each thread operates on independent data (separate blocks, separate output arrays).
