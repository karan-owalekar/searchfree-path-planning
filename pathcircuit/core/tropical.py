from __future__ import annotations

import numpy as np

SENTINEL = 536_870_912  # 2^29


def tropical_add(a: int, b: int) -> int:
    if a >= SENTINEL or b >= SENTINEL:
        return SENTINEL
    s = a + b
    if s > SENTINEL or s < 0:
        return SENTINEL
    return s


def tropical_min(a: int, b: int) -> int:
    return a if a <= b else b


# ---------------------------------------------------------------------------
# Numpy-accelerated matrix operations
# ---------------------------------------------------------------------------

def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C[i,j] = min_k (A[i,k] + B[k,j]) with SENTINEL clamping."""
    # A: (n, m)  B: (m, p)  -> C: (n, p)
    # Use int64 intermediates to avoid overflow during addition
    a = A.astype(np.int64)
    b = B.astype(np.int64)
    sums = a[:, :, np.newaxis] + b[np.newaxis, :, :]  # (n, m, p)
    sums = np.clip(sums, 0, SENTINEL)
    C = np.min(sums, axis=1).astype(np.int32)  # (n, p)
    return C


def tropical_matmul_with_witness(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C, W) where C is the tropical product and W[i,j] = argmin_k."""
    a = A.astype(np.int64)
    b = B.astype(np.int64)
    sums = a[:, :, np.newaxis] + b[np.newaxis, :, :]  # (n, m, p)
    sums = np.clip(sums, 0, SENTINEL)
    W = np.argmin(sums, axis=1).astype(np.int32)  # (n, p)
    C = np.min(sums, axis=1).astype(np.int32)      # (n, p)
    return C, W
