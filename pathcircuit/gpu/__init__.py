"""GPU acceleration for path planning via CUDA (requires CuPy + working NVIDIA driver)."""
from __future__ import annotations

import ctypes as _ctypes
import os as _os


def _setup_cuda_env() -> None:
    """Pre-load pip-installed NVIDIA libs so CuPy can find them via dlopen."""
    try:
        import nvidia.cuda_runtime as _cr
        base = _os.path.dirname(_os.path.dirname(_cr.__file__))
    except ImportError:
        return

    # Preload shared libraries with RTLD_GLOBAL so CuPy's dlopen finds them
    for lib in (
        _os.path.join(base, "cuda_nvrtc", "lib", "libnvrtc.so.12"),
        _os.path.join(base, "cuda_nvrtc", "lib", "libnvrtc-builtins.so.12.9"),
    ):
        if _os.path.isfile(lib):
            try:
                _ctypes.CDLL(lib, mode=_ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    rt_dir = _os.path.join(base, "cuda_runtime")
    if _os.path.isdir(rt_dir) and "CUDA_PATH" not in _os.environ:
        _os.environ["CUDA_PATH"] = rt_dir


_setup_cuda_env()


def is_gpu_available() -> bool:
    """Check whether CUDA compute is actually functional (not just installed)."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        a = cp.zeros(1, dtype=cp.int32)
        del a
        return True
    except Exception:
        return False
