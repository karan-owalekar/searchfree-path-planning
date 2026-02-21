import argparse
import sys
import time

from pathcircuit.warmup import warmup_jit


def main():
    parser = argparse.ArgumentParser(description="Search-Free Path Planner")
    parser.add_argument(
        "--compare-algorithms",
        action="store_true",
        help="Print timing comparison with Dijkstra and A* on each path query",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    warmup_jit()
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"  JIT ready ({warmup_ms:.0f} ms)", file=sys.stderr)

    from pathcircuit.ui.app import App
    app = App(compare=args.compare_algorithms)
    app.run()


if __name__ == "__main__":
    main()
