# Search-Free Path Planning

An algebraic approach to shortest-path computation on grid graphs. Instead of iterative search (Dijkstra, A\*), this project computes shortest paths through **hierarchical tropical composition** of local block summaries — a fixed, feed-forward pipeline with no data-dependent branching.

## The Idea

Traditional pathfinding algorithms explore the graph node-by-node, making them inherently sequential and hard to parallelize. This project takes a fundamentally different approach:

1. **Local Summarization** — The grid is divided into 4×4 tiles. Each tile is summarized by a *transfer matrix* encoding shortest distances between all boundary cells using tropical (min-plus) matrix repeated squaring.

2. **Hierarchical Composition** — Adjacent blocks are merged bottom-up in a quadtree using tropical matrix multiplication and the *tropical Schur complement*, algebraically eliminating interior cells at each level.

3. **Search-Free Queries** — Once the hierarchy is built, any point-to-point distance query is answered via flat table lookups — no priority queues, no graph traversal.

The algorithm produces **exact** results (bitwise-identical to Dijkstra) using integer arithmetic throughout.

## Performance

The planner offers two optimization profiles via `--optimize`, and optional GPU acceleration via `--gpu`:

- **`--optimize path`** (default) — Builds a linked-list successor table and intra-block path atlas during oracle construction. Path reconstruction is ~2–3μs (JIT kernel), making it faster than A\* on all grid sizes.
- **`--optimize oracle`** — Skips the successor table for a faster oracle build. Path reconstruction falls back to an inline greedy walk (~60–400μs).
- **`--gpu`** — Accelerates the boundary table construction with a CUDA shared-memory kernel, roughly halving oracle build time on supported hardware.

Distance queries are equally fast (~1–2μs) in both modes.

### Benchmark (64×64 grid, 8-connected, ~500 obstacles)

| Algorithm | Build | Distance Query | Path Reconstruction | Total Query |
|---|---|---|---|---|
| **Search-Free (`--optimize path`, GPU)** | ~170 ms | 0.013 ms | 0.053 ms | **0.066 ms** |
| **Search-Free (`--optimize path`, CPU)** | ~355 ms | 0.030 ms | 0.063 ms | **0.093 ms** |
| **Search-Free (`--optimize oracle`, CPU)** | ~310 ms | 0.030 ms | 0.287 ms | 0.317 ms |
| A\* | — | — | — | 0.06–0.19 ms |
| Dijkstra | — | — | — | 0.18–0.30 ms |

With `--optimize path`, the search-free planner matches or beats A\* on total query time while providing O(1) distance lookups. The advantage grows on larger grids since A\*/Dijkstra scale with search difficulty while the oracle query stays nearly constant.

**Amortization**: the oracle build cost is paid once per map. After ~5 queries the path-optimized oracle breaks even with A\*; after ~3 queries the oracle-optimized variant breaks even with Dijkstra.

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/searchfree-path-planning.git
cd searchfree-path-planning
```

**2. Install uv (if not already installed)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Create the virtual environment and install dependencies**

```bash
uv sync
```

This reads `pyproject.toml`, creates a `.venv`, pins Python 3.9, and installs all dependencies.

**4. Run the application**

```bash
uv run python main.py
```

### Command-Line Options

| Flag | Description |
|---|---|
| `--compare-algorithms` | Print timing comparison with Dijkstra and A\* on each path query |
| `--optimize oracle` | Optimize for fast oracle build (skip successor table) |
| `--optimize path` | Optimize for fast path queries (default; builds successor table + atlas) |
| `--gpu` | Use GPU (CUDA) for oracle build acceleration and Dijkstra comparison |

Examples:

```bash
# Default: fast path queries
uv run python main.py

# Side-by-side comparison with A* and Dijkstra
uv run python main.py --compare-algorithms

# Fast oracle rebuild (useful for frequently changing maps)
uv run python main.py --optimize oracle --compare-algorithms

# GPU-accelerated mode (requires NVIDIA GPU + CUDA)
uv run python main.py --gpu --compare-algorithms
```

## Usage

The application opens an interactive pygame window where you can draw grids, place obstacles, and visualize paths.

| Action | Control |
|---|---|
| Draw / erase obstacles | Left-click and drag on the grid |
| Set **start** point | Press `S`, then click a cell |
| Set **goal** point | Press `G`, then click a cell |
| Change grid size | Use the size dropdown (8×8 to 64×64) |
| Toggle connectivity | Switch between 4-connected and 8-connected grids |
| Find path | Click **Find Path** (builds oracle on first use, then queries it) |
| View instructions | Click **Info** |

Once both start and goal are placed, the oracle is built and the shortest path is displayed. With `--compare-algorithms`, timing breakdowns are printed to the terminal:

```
  ┌─ Search-Free Planner ─────────────────────┐
  │  Oracle Build                 354.70 ms   │
  │  Distance Query                0.030 ms   │
  │  Recon                         0.063 ms   │
  │  Total Query                   0.093 ms   │
  │  Path Cost                          348   │
  │  Path Length                  175 cells   │
  └───────────────────────────────────────────┘

  ┌─ A* (CPU) ────────────────────────────────┐
  │  Time                           0.15 ms   │
  │  Nodes Expanded                    1131   │
  │  Path Cost                          348   │
  │  Path Length                  175 cells   │
  └───────────────────────────────────────────┘

  ┌─ Dijkstra (CPU) ──────────────────────────┐
  │  Time                           0.24 ms   │
  │  Nodes Expanded                    2431   │
  │  Path Cost                          348   │
  │  Path Length                  175 cells   │
  └───────────────────────────────────────────┘
```

## Project Structure

```
searchfree-path-planning/
├── main.py                    # Entry point (--compare-algorithms, --optimize, --gpu)
├── pathcircuit/
│   ├── grid.py                # Grid representation and obstacle handling
│   ├── warmup.py              # Eager JIT / GPU kernel compilation
│   ├── core/
│   │   ├── tropical.py        # Tropical semiring operations (min-plus)
│   │   ├── base_block.py      # Phase 1: 4×4 base block transfer matrices
│   │   ├── hierarchy.py       # Phase 2: Bottom-up quadtree composition
│   │   ├── uncondense.py      # Phase 3a: Top-down distance recovery
│   │   ├── oracle.py          # Oracle, linked-list retrieval, JIT kernels
│   │   └── pipeline.py        # Orchestrates all build phases
│   ├── compare/
│   │   ├── runner.py          # Algorithm comparison harness
│   │   ├── astar.py           # A* baseline (Numba JIT)
│   │   └── dijkstra.py        # Dijkstra baseline (Numba JIT)
│   ├── gpu/
│   │   ├── __init__.py        # CUDA environment setup
│   │   ├── boundary_gpu.py    # GPU boundary table build (CuPy shared-memory kernel)
│   │   ├── oracle_gpu.py      # GPU path reconstruction (CuPy CUDA kernels)
│   │   └── dijkstra_gpu.py    # GPU Dijkstra via parallel Bellman-Ford
│   └── ui/
│       ├── app.py             # Pygame application loop
│       ├── controls.py        # Buttons, dropdowns, status bar
│       ├── theme.py           # Visual constants
│       └── grid_view.py       # Grid rendering
└── idea/                      # Design documents and theory
```

## Architecture

### Oracle Build

1. **Base blocks** — Each 4×4 tile computes an all-pairs shortest path matrix via tropical repeated squaring (4 rounds). Parallelized across blocks with `ThreadPoolExecutor` + `nogil` JIT kernels.
2. **Hierarchy** — Blocks merge bottom-up in a quadtree. Each merge uses tropical Schur complement to algebraically eliminate interior boundary cells. Joint matrix assembly is JIT-compiled.
3. **Un-condensation** — Top-down pass recovers global distances for every boundary cell.
4. **Flat boundary table** — All boundary-to-boundary distances are extracted into a single `B×B` lookup table.
   - *CPU*: parallelized with `ThreadPoolExecutor` + `nogil` Numba kernels, pre-allocated scratch buffers.
   - *GPU* (`--gpu`): a CUDA shared-memory kernel (one block per source boundary cell, cooperative threads with two-pass gather-then-scatter expansion) via CuPy, roughly halving build time.
5. **Linked-list structures** (`--optimize path` only):
   - *Intra-block path atlas* — precomputed cell-by-cell paths between all boundary pairs within each block (~324 KB for 64×64).
   - *Boundary successor table* — `B×B` table where `successor[target, current]` gives the next boundary hop toward the target (built in parallel with `numba.prange`).

### Query

- **Distance**: 144 flat table lookups (12×12 boundary pairs), ~1–2μs.
- **Path** (`--optimize path`): Gateway selection → first-mile walk → successor pointer chasing with atlas reads → last-mile walk. The JIT kernel runs in ~2–3μs. Used even in `--gpu` mode since it outperforms GPU reconstruction.
- **Path** (`--optimize oracle`): Inline greedy walk computing `d(neighbor, goal)` on-the-fly via 144 lookups per step. Optionally GPU-accelerated via a successor-field CUDA kernel.
