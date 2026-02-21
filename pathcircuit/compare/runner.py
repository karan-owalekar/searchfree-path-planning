from __future__ import annotations

from pathcircuit.grid import Grid, SENTINEL
from pathcircuit.compare.dijkstra import dijkstra
from pathcircuit.compare.astar import astar

BOX_WIDTH = 45


def _box(title: str, rows: list[tuple[str, str]]) -> str:
    inner = BOX_WIDTH - 2
    lines: list[str] = []
    header = f" {title} "
    lines.append(f"  \u250c\u2500{header}\u2500" + "\u2500" * (inner - len(header) - 2) + "\u2510")
    for label, value in rows:
        content = f"  {label:<20s}{value:>18s}"
        # Pad or truncate to fit
        content = content[:inner]
        content = content.ljust(inner)
        lines.append(f"  \u2502{content}\u2502")
    lines.append(f"  \u2514" + "\u2500" * inner + "\u2518")
    return "\n".join(lines)


def run_comparison(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    oracle_build_ms: float,
    oracle_dist_ms: float,
    oracle_recon_ms: float,
    oracle_path: list[tuple[int, int]],
    oracle_cost: int,
    use_gpu: bool = False,
) -> None:
    """Run Dijkstra and A*, then print a formatted comparison to stdout."""
    obstacle_count = int((grid.costs >= SENTINEL).sum())
    mode_tag = " (GPU)" if use_gpu else ""

    print()
    print("  " + "=" * (BOX_WIDTH - 2))
    print(f"  {'ALGORITHM  COMPARISON':^{BOX_WIDTH - 2}s}")
    print("  " + "=" * (BOX_WIDTH - 2))
    info = (
        f"  Grid: {grid.rows}\u00d7{grid.cols}  "
        f"Neighbors: {grid.connectivity}  "
        f"Obstacles: {obstacle_count}"
    )
    print(info)
    print(f"  Start: {start}  Goal: {goal}")
    if use_gpu:
        print("  Mode: GPU (CUDA)")
    print()

    cost_str = str(oracle_cost) if oracle_cost < SENTINEL else "unreachable"
    path_len = str(len(oracle_path)) + " cells" if oracle_path else "N/A"
    query_ms = oracle_dist_ms + oracle_recon_ms
    recon_label = "Recon" + mode_tag
    print(_box("Search-Free Planner", [
        ("Oracle Build", f"{oracle_build_ms:.2f} ms"),
        ("Distance Query", f"{oracle_dist_ms:.3f} ms"),
        (recon_label, f"{oracle_recon_ms:.3f} ms"),
        ("Total Query", f"{query_ms:.3f} ms"),
        ("Path Cost", cost_str),
        ("Path Length", path_len),
    ]))
    print()

    if use_gpu:
        from pathcircuit.gpu.dijkstra_gpu import gpu_dijkstra
        gpu_res = gpu_dijkstra(grid, start, goal)

        cost_str = str(gpu_res.cost) if gpu_res.cost < SENTINEL else "unreachable"
        path_len = str(len(gpu_res.path)) + " cells" if gpu_res.path else "N/A"
        print(_box("GPU Dijkstra (Bellman-Ford)", [
            ("Time", f"{gpu_res.time_ms:.2f} ms"),
            ("Iterations", str(gpu_res.iterations)),
            ("Path Cost", cost_str),
            ("Path Length", path_len),
        ]))
        print()

    # CPU baselines always shown for reference
    dijk = dijkstra(grid, start, goal)
    astar_res = astar(grid, start, goal)

    cost_str = str(astar_res.cost) if astar_res.cost < SENTINEL else "unreachable"
    path_len = str(len(astar_res.path)) + " cells" if astar_res.path else "N/A"
    print(_box("A* (CPU)", [
        ("Time", f"{astar_res.time_ms:.2f} ms"),
        ("Nodes Expanded", str(astar_res.nodes_expanded)),
        ("Path Cost", cost_str),
        ("Path Length", path_len),
    ]))
    print()

    cost_str = str(dijk.cost) if dijk.cost < SENTINEL else "unreachable"
    path_len = str(len(dijk.path)) + " cells" if dijk.path else "N/A"
    print(_box("Dijkstra (CPU)", [
        ("Time", f"{dijk.time_ms:.2f} ms"),
        ("Nodes Expanded", str(dijk.nodes_expanded)),
        ("Path Cost", cost_str),
        ("Path Length", path_len),
    ]))

    print()
    print("  " + "=" * (BOX_WIDTH - 2))
    print()
