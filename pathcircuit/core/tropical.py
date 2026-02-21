from __future__ import annotations

import numpy as np

SENTINEL = 536_870_912  # 2^29

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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
# Numba-accelerated kernels (compiled once, cached to disk)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @_njit(cache=True, nogil=True)
    def _matmul_jit(A: np.ndarray, B: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        _S = np.int64(536_870_912)
        n, m = A.shape
        p = B.shape[1]
        C = np.empty((n, p), dtype=np.int32)
        for i in range(n):
            for j in range(p):
                best = _S
                for k in range(m):
                    a_val = np.int64(A[i, k])
                    if a_val >= _S:
                        continue
                    b_val = np.int64(B[k, j])
                    if b_val >= _S:
                        continue
                    s = a_val + b_val
                    if s < best:
                        best = s
                C[i, j] = np.int32(best)
        return C

    @_njit(cache=True, nogil=True)
    def _matmul_witness_jit(A: np.ndarray, B: np.ndarray) -> tuple:  # type: ignore[misc]
        _S = np.int64(536_870_912)
        n, m = A.shape
        p = B.shape[1]
        C = np.empty((n, p), dtype=np.int32)
        W = np.empty((n, p), dtype=np.int32)
        for i in range(n):
            for j in range(p):
                best = _S
                best_k = np.int32(0)
                for k in range(m):
                    a_val = np.int64(A[i, k])
                    if a_val >= _S:
                        continue
                    b_val = np.int64(B[k, j])
                    if b_val >= _S:
                        continue
                    s = a_val + b_val
                    if s < best:
                        best = s
                        best_k = np.int32(k)
                C[i, j] = np.int32(best)
                W[i, j] = best_k
        return C, W

    @_njit(cache=True, nogil=True)
    def minplus_mv(D_global: np.ndarray, bridge_idx: np.ndarray,  # type: ignore[misc]
                   bridge_d: np.ndarray, n_cells: int) -> np.ndarray:
        """d_new[k] = min_b( bridge_d[b] + D_global[bridge_idx[b], k] )"""
        _S = np.int64(536_870_912)
        n_bridges = len(bridge_idx)
        result = np.empty(n_cells, dtype=np.int64)
        for k in range(n_cells):
            best = _S
            for b in range(n_bridges):
                dg = np.int64(D_global[bridge_idx[b], k])
                if dg >= _S:
                    continue
                bd = bridge_d[b]
                if bd >= _S:
                    continue
                s = bd + dg
                if s < best:
                    best = s
            result[k] = best
        return result

else:  # pure-numpy fallback
    def _matmul_jit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return None  # type: ignore[return-value]  # unused

    def _matmul_witness_jit(A: np.ndarray, B: np.ndarray) -> tuple:
        return None, None  # type: ignore[return-value]

    def minplus_mv(D_global: np.ndarray, bridge_idx: np.ndarray,
                   bridge_d: np.ndarray, n_cells: int) -> np.ndarray:
        D_sub = D_global[bridge_idx, :].astype(np.int64)
        sums = bridge_d[:, np.newaxis] + D_sub
        np.clip(sums, 0, SENTINEL, out=sums)
        return np.min(sums, axis=0)


# ---------------------------------------------------------------------------
# Public matrix operations (dispatch to JIT when available)
# ---------------------------------------------------------------------------

def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C[i,j] = min_k (A[i,k] + B[k,j]) with SENTINEL clamping."""
    if _HAS_NUMBA:
        return _matmul_jit(A, B)
    a = A.astype(np.int64)
    b = B.astype(np.int64)
    sums = a[:, :, np.newaxis] + b[np.newaxis, :, :]
    sums = np.clip(sums, 0, SENTINEL)
    return np.min(sums, axis=1).astype(np.int32)


def tropical_matmul_with_witness(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C, W) where C is the tropical product and W[i,j] = argmin_k."""
    if _HAS_NUMBA:
        return _matmul_witness_jit(A, B)
    a = A.astype(np.int64)
    b = B.astype(np.int64)
    sums = a[:, :, np.newaxis] + b[np.newaxis, :, :]
    sums = np.clip(sums, 0, SENTINEL)
    W = np.argmin(sums, axis=1).astype(np.int32)
    C = np.min(sums, axis=1).astype(np.int32)
    return C, W
