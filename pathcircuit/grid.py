from __future__ import annotations

import math
import numpy as np

SENTINEL = 536_870_912  # 2^29

CARDINAL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class Grid:
    """Rectangular grid with integer cell costs and configurable connectivity."""

    def __init__(self, rows: int, cols: int, connectivity: int = 8):
        assert connectivity in (4, 8)
        self.rows = rows
        self.cols = cols
        self.connectivity = connectivity
        self.costs = np.ones((rows, cols), dtype=np.int32)

    # -- obstacle helpers --

    def set_obstacle(self, r: int, c: int) -> None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.costs[r, c] = SENTINEL

    def clear_obstacle(self, r: int, c: int) -> None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.costs[r, c] = 1

    def is_obstacle(self, r: int, c: int) -> bool:
        return self.costs[r, c] >= SENTINEL

    # -- neighbor enumeration --

    def get_neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        offsets = CARDINAL_OFFSETS
        if self.connectivity == 8:
            offsets = CARDINAL_OFFSETS + DIAGONAL_OFFSETS
        result: list[tuple[int, int]] = []
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append((nr, nc))
        return result

    def edge_weight(self, r1: int, c1: int, r2: int, c2: int) -> int:
        ci = int(self.costs[r1, c1])
        cj = int(self.costs[r2, c2])
        if ci >= SENTINEL or cj >= SENTINEL:
            return SENTINEL
        return ci + cj

    # -- padding for the quadtree hierarchy --

    @staticmethod
    def _next_valid_dim(d: int) -> int:
        """Return the smallest n >= d where n is a multiple of 4 and n/4 is a power of 2."""
        if d <= 4:
            return 4
        blocks = math.ceil(d / 4)
        power = 1
        while power < blocks:
            power *= 2
        return power * 4

    def padded_size(self) -> int:
        return max(self._next_valid_dim(self.rows), self._next_valid_dim(self.cols))

    def to_padded(self) -> Grid:
        n = self.padded_size()
        if n == self.rows == self.cols:
            return self
        padded = Grid(n, n, self.connectivity)
        padded.costs[:] = SENTINEL
        padded.costs[: self.rows, : self.cols] = self.costs
        return padded

    def copy(self) -> Grid:
        g = Grid(self.rows, self.cols, self.connectivity)
        g.costs = self.costs.copy()
        return g
