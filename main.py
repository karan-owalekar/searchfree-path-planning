import argparse

from pathcircuit.ui.app import App


def main():
    parser = argparse.ArgumentParser(description="Search-Free Path Planner")
    parser.add_argument(
        "--compare-algorithms",
        action="store_true",
        help="Print timing comparison with Dijkstra and A* on each path query",
    )
    args = parser.parse_args()

    app = App(compare=args.compare_algorithms)
    app.run()


if __name__ == "__main__":
    main()
