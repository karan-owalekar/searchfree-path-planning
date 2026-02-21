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
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (CUDA) for oracle reconstruction and comparison algorithms",
    )
    parser.add_argument(
        "--optimize",
        choices=["oracle", "path"],
        default="path",
        help="Optimize for fast oracle build or fast path query (default: path)",
    )
    args = parser.parse_args()

    print(f"  Optimize: {args.optimize}", file=sys.stderr)

    if args.gpu:
        from pathcircuit.gpu import is_gpu_available
        if not is_gpu_available():
            print(
                "  ERROR: --gpu requested but CUDA is not available.\n"
                "  Possible fixes:\n"
                "    1. Install proprietary NVIDIA driver: sudo apt install nvidia-driver-570\n"
                "    2. Reboot after driver install\n"
                "    3. Check nvidia-smi works\n",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  GPU mode enabled (CUDA)", file=sys.stderr)

    t0 = time.perf_counter()
    warmup_jit()
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"  JIT ready ({warmup_ms:.0f} ms)", file=sys.stderr)

    if args.gpu:
        from pathcircuit.warmup import warmup_gpu
        t0 = time.perf_counter()
        warmup_gpu()
        gpu_ms = (time.perf_counter() - t0) * 1000
        print(f"  GPU kernels ready ({gpu_ms:.0f} ms)", file=sys.stderr)

    from pathcircuit.ui.app import App
    app = App(compare=args.compare_algorithms, use_gpu=args.gpu,
              optimize=args.optimize)
    app.run()


if __name__ == "__main__":
    main()
