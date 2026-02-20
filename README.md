# Search-Free Path Planning

An algebraic approach to shortest-path computation on grid graphs. Instead of iterative search (Dijkstra, A\*), this project computes shortest paths through **hierarchical tropical composition** of local block summaries — a fixed, feed-forward pipeline with no data-dependent branching.

## The Idea

Traditional pathfinding algorithms explore the graph node-by-node, making them inherently sequential and hard to parallelize. This project takes a fundamentally different approach:

1. **Local Summarization** — The grid is divided into 4×4 tiles. Each tile is summarized by a *transfer matrix* encoding shortest distances between all boundary cells using tropical (min-plus) matrix repeated squaring.

2. **Hierarchical Composition** — Adjacent blocks are merged bottom-up in a quadtree using tropical matrix multiplication and the *tropical Schur complement*, algebraically eliminating interior cells at each level.

3. **Search-Free Queries** — Once the hierarchy is built, any point-to-point distance query is answered in O(log N) time via min-plus matrix–vector products — no priority queues, no graph traversal.

The algorithm produces **exact** results (bitwise-identical to Dijkstra) using integer arithmetic throughout.

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

This reads `pyproject.toml`, creates a `.venv`, pins Python 3.9, and installs `pygame-ce` and `numpy`.

**4. Run the application**

```bash
uv run python main.py
```

To enable side-by-side timing comparison with Dijkstra and A\*:

```bash
uv run python main.py --compare-algorithms
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

Once both start and goal are placed, the oracle is built and the shortest path is displayed. With `--compare-algorithms`, timing breakdowns for the algebraic oracle vs. Dijkstra and A\* are printed to the terminal.

## Project Structure

```
searchfree-path-planning/
├── main.py                    # Entry point
├── pathcircuit/
│   ├── grid.py                # Grid representation and obstacle handling
│   ├── core/
│   │   ├── tropical.py        # Tropical semiring operations (min-plus)
│   │   ├── base_block.py      # Phase 1: 4×4 base block transfer matrices
│   │   ├── hierarchy.py       # Phase 2: Bottom-up quadtree composition
│   │   ├── uncondense.py      # Phase 3a: Top-down distance recovery
│   │   ├── oracle.py          # Phase 3b: Query system and path reconstruction
│   │   └── pipeline.py        # Orchestrates all phases
│   ├── compare/
│   │   ├── runner.py          # Algorithm comparison harness
│   │   ├── astar.py           # A* baseline
│   │   └── dijkstra.py        # Dijkstra baseline
│   └── ui/
│       ├── app.py             # Pygame application loop
│       ├── controls.py        # Buttons, dropdowns, status bar
│       ├── theme.py           # Visual constants
│       └── grid_view.py       # Grid rendering
└── idea/                      # Design documents and theory
```
