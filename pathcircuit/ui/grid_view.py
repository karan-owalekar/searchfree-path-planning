from __future__ import annotations

import pygame

from pathcircuit.grid import Grid
from pathcircuit.ui import theme


class GridView:
    """Renders the grid and translates pixel coords to cell coords."""

    def __init__(self, grid: Grid, area: pygame.Rect):
        self.grid = grid
        self.area = area
        self.start: tuple[int, int] | None = None
        self.goal: tuple[int, int] | None = None
        self.path: list[tuple[int, int]] = []
        self._cell_size = theme.DEFAULT_CELL_SIZE
        self._recompute_layout()

    def set_grid(self, grid: Grid) -> None:
        self.grid = grid
        self.start = None
        self.goal = None
        self.path = []
        self._recompute_layout()

    def set_area(self, area: pygame.Rect) -> None:
        self.area = area
        self._recompute_layout()

    def _recompute_layout(self) -> None:
        max_cell_w = (self.area.w - theme.CELL_PADDING) // self.grid.cols - theme.CELL_PADDING
        max_cell_h = (self.area.h - theme.CELL_PADDING) // self.grid.rows - theme.CELL_PADDING
        self._cell_size = max(theme.MIN_CELL_SIZE, min(max_cell_w, max_cell_h))

        total_w = self.grid.cols * (self._cell_size + theme.CELL_PADDING) + theme.CELL_PADDING
        total_h = self.grid.rows * (self._cell_size + theme.CELL_PADDING) + theme.CELL_PADDING
        self._origin_x = self.area.x + (self.area.w - total_w) // 2
        self._origin_y = self.area.y + (self.area.h - total_h) // 2

    def pixel_to_cell(self, px: int, py: int) -> tuple[int, int] | None:
        rx = px - self._origin_x - theme.CELL_PADDING
        ry = py - self._origin_y - theme.CELL_PADDING
        if rx < 0 or ry < 0:
            return None
        step = self._cell_size + theme.CELL_PADDING
        col = rx // step
        row = ry // step
        # Check the click is inside the cell, not in the padding gap
        if rx % step >= self._cell_size or ry % step >= self._cell_size:
            return None
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            return (int(row), int(col))
        return None

    def draw(self, surface: pygame.Surface) -> None:
        path_set = set(self.path)
        step = self._cell_size + theme.CELL_PADDING
        cs = self._cell_size

        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                x = self._origin_x + theme.CELL_PADDING + c * step
                y = self._origin_y + theme.CELL_PADDING + r * step
                rect = pygame.Rect(x, y, cs, cs)

                if (r, c) == self.start:
                    color = theme.CELL_START
                elif (r, c) == self.goal:
                    color = theme.CELL_GOAL
                elif (r, c) in path_set and (r, c) != self.start and (r, c) != self.goal:
                    color = theme.CELL_PATH
                elif self.grid.is_obstacle(r, c):
                    color = theme.CELL_OBSTACLE
                else:
                    color = theme.CELL_EMPTY

                pygame.draw.rect(surface, color, rect, border_radius=2)
