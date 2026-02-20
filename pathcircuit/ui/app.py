from __future__ import annotations

import time

import pygame

from pathcircuit.grid import Grid, SENTINEL
from pathcircuit.core.pipeline import build_pathcircuit_oracle
from pathcircuit.core.oracle import Oracle, query_distance, reconstruct_path
from pathcircuit.compare.runner import run_comparison
from pathcircuit.ui import theme
from pathcircuit.ui.grid_view import GridView
from pathcircuit.ui.controls import Button, Dropdown, StatusBar, InfoOverlay

GRID_PRESETS = ["8\u00d78", "16\u00d716", "32\u00d732", "64\u00d764"]
GRID_SIZES = [8, 16, 32, 64]
NEIGHBOR_OPTIONS = ["8", "4"]


class App:
    def __init__(self, compare: bool = False):
        pygame.init()
        self._compare = compare

        info = pygame.display.Info()
        w = max(theme.WINDOW_MIN_W, int(info.current_w * 0.75))
        h = max(theme.WINDOW_MIN_H, int(info.current_h * 0.82))
        self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption("Search-Free Path Planner")

        self._font = pygame.font.SysFont(theme.FONT_NAME, theme.FONT_SIZE_NORMAL)
        self._font_sm = pygame.font.SysFont(theme.FONT_NAME, theme.FONT_SIZE_SMALL)
        self._font_title = pygame.font.SysFont(theme.FONT_NAME, theme.FONT_SIZE_TITLE, bold=True)

        self._grid = Grid(16, 16, connectivity=8)
        grid_area = self._grid_area()
        self._grid_view = GridView(self._grid, grid_area)

        self._build_controls()
        self._status = StatusBar(h - theme.STATUS_HEIGHT, w)
        self._info_overlay = InfoOverlay()

        self._oracle: Oracle | None = None
        self._oracle_dirty = True
        self._oracle_build_ms = 0.0

        self._mode = "draw"  # "draw" | "set_start" | "set_goal"
        self._drawing = False
        self._draw_obstacle = True  # True = adding, False = erasing

        self._clock = pygame.time.Clock()

    # ---- layout helpers ----

    def _grid_area(self) -> pygame.Rect:
        w, h = self._screen.get_size()
        return pygame.Rect(0, theme.TOOLBAR_HEIGHT, w, h - theme.TOOLBAR_HEIGHT - theme.STATUS_HEIGHT)

    def _build_controls(self) -> None:
        y = 18
        h = 60

        self._dd_grid = Dropdown(100, y, 150, h, "Grid", GRID_PRESETS, selected=1)
        self._dd_neighbors = Dropdown(460, y, 120, h, "Neighbors", NEIGHBOR_OPTIONS, selected=0)
        self._btn_find = Button(610, y, 180, h, "Find Path", accent=True)
        self._btn_clear_path = Button(806, y, 170, h, "Clear Path")
        self._btn_clear_grid = Button(992, y, 170, h, "Clear Grid")
        self._btn_info = Button(1178, y, h, h, "?")

    # ---- event handling ----

    def _handle_dropdown_change(self) -> None:
        size = GRID_SIZES[self._dd_grid.selected]
        conn = int(self._dd_neighbors.value)
        if size != self._grid.rows or size != self._grid.cols or conn != self._grid.connectivity:
            self._grid = Grid(size, size, connectivity=conn)
            self._grid_view.set_grid(self._grid)
            self._oracle = None
            self._oracle_dirty = True
            self._status.text = f"Grid reset to {size}\u00d7{size}, {conn}-connected"

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.VIDEORESIZE:
                w, h = event.w, event.h
                self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                self._grid_view.set_area(self._grid_area())
                self._status.update_rect(h - theme.STATUS_HEIGHT, w)

            # Info overlay eats all input while visible
            if self._info_overlay.handle_event(event):
                continue

            # Dropdowns (handle first so they can consume clicks when open)
            any_dd_open = self._dd_grid.is_open or self._dd_neighbors.is_open
            if self._dd_grid.handle_event(event) or self._dd_neighbors.handle_event(event):
                self._handle_dropdown_change()
                continue
            if any_dd_open:
                continue  # swallow event while dropdown was open

            # Buttons
            if self._btn_find.handle_event(event):
                self._find_path()
                continue
            if self._btn_clear_path.handle_event(event):
                self._grid_view.path = []
                self._status.text = "Path cleared"
                continue
            if self._btn_clear_grid.handle_event(event):
                self._grid = Grid(self._grid.rows, self._grid.cols, self._grid.connectivity)
                self._grid_view.set_grid(self._grid)
                self._oracle = None
                self._oracle_dirty = True
                self._status.text = "Grid cleared"
                continue
            if self._btn_info.handle_event(event):
                self._info_overlay.toggle()
                continue

            # Keyboard
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self._mode = "set_start"
                    self._status.text = "Click a cell to set START"
                    continue
                if event.key == pygame.K_g:
                    self._mode = "set_goal"
                    self._status.text = "Click a cell to set GOAL"
                    continue

            # Mouse interaction with the grid
            if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                cell = self._grid_view.pixel_to_cell(*event.pos)
                if cell is None:
                    continue

                if self._mode == "set_start":
                    if not self._grid.is_obstacle(*cell):
                        self._grid_view.start = cell
                        self._grid_view.path = []
                        self._status.text = f"Start set to {cell}"
                    self._mode = "draw"
                    continue
                if self._mode == "set_goal":
                    if not self._grid.is_obstacle(*cell):
                        self._grid_view.goal = cell
                        self._grid_view.path = []
                        self._status.text = f"Goal set to {cell}"
                    self._mode = "draw"
                    continue

                # Drawing/erasing mode
                self._drawing = True
                if event.button == 1:
                    self._draw_obstacle = not self._grid.is_obstacle(*cell)
                else:
                    self._draw_obstacle = False
                self._apply_draw(cell)

            if event.type == pygame.MOUSEBUTTONUP:
                self._drawing = False

            if event.type == pygame.MOUSEMOTION and self._drawing:
                cell = self._grid_view.pixel_to_cell(*event.pos)
                if cell is not None:
                    self._apply_draw(cell)

            # Hover for buttons
            self._btn_find.handle_event(event)
            self._btn_clear_path.handle_event(event)
            self._btn_clear_grid.handle_event(event)
            self._btn_info.handle_event(event)

        return True

    def _apply_draw(self, cell: tuple[int, int]) -> None:
        # Don't overwrite start/goal
        if cell == self._grid_view.start or cell == self._grid_view.goal:
            return
        if self._draw_obstacle:
            self._grid.set_obstacle(*cell)
        else:
            self._grid.clear_obstacle(*cell)
        self._oracle_dirty = True
        self._grid_view.path = []

    # ---- pathfinding ----

    def _find_path(self) -> None:
        start = self._grid_view.start
        goal = self._grid_view.goal
        if start is None or goal is None:
            self._status.text = "Set start (S) and goal (G) first"
            return

        # Build oracle if needed
        if self._oracle is None or self._oracle_dirty:
            self._status.text = "Building oracle..."
            pygame.display.flip()
            t0 = time.perf_counter()
            self._oracle = build_pathcircuit_oracle(self._grid)
            self._oracle_build_ms = (time.perf_counter() - t0) * 1000.0
            self._oracle_dirty = False
            print(f"\n  Oracle built in {self._oracle_build_ms:.2f} ms")

        t0 = time.perf_counter()
        dist = query_distance(self._oracle, start, goal)
        path = reconstruct_path(self._oracle, start, goal)
        query_ms = (time.perf_counter() - t0) * 1000.0

        if dist >= SENTINEL:
            self._grid_view.path = []
            self._status.text = "No path found (unreachable)"
            print(f"  Query: unreachable  ({query_ms:.2f} ms)")
        else:
            self._grid_view.path = path
            self._status.text = f"Path found \u2014 cost: {dist}  length: {len(path)} cells  ({query_ms:.2f} ms)"
            print(f"  Query: cost={dist}  length={len(path)}  ({query_ms:.2f} ms)")

        if self._compare:
            run_comparison(
                self._grid, start, goal,
                self._oracle_build_ms, query_ms,
                path, dist,
            )

    # ---- rendering ----

    def _draw(self) -> None:
        self._screen.fill(theme.BG)

        # Toolbar background
        w = self._screen.get_width()
        pygame.draw.rect(self._screen, theme.PANEL_BG, (0, 0, w, theme.TOOLBAR_HEIGHT))

        # Title (only when window is wide enough that it won't overlap controls)
        title = self._font_title.render("Search-Free Path Planner", True, theme.TITLE_TEXT)
        title_x = w - title.get_width() - 20
        if title_x > 1260:
            self._screen.blit(title, (title_x, (theme.TOOLBAR_HEIGHT - title.get_height()) // 2))

        # Grid (drawn before controls so open dropdown lists aren't hidden)
        self._grid_view.draw(self._screen)

        # Controls
        self._dd_grid.draw(self._screen, self._font)
        self._dd_neighbors.draw(self._screen, self._font)
        self._btn_find.draw(self._screen, self._font)
        self._btn_clear_path.draw(self._screen, self._font)
        self._btn_clear_grid.draw(self._screen, self._font)
        self._btn_info.draw(self._screen, self._font)

        # Status bar
        self._status.draw(self._screen, self._font_sm)

        # Info overlay (drawn last, on top of everything)
        self._info_overlay.draw(self._screen, self._font, self._font_sm, self._font_title)

        pygame.display.flip()

    # ---- main loop ----

    def run(self) -> None:
        running = True
        while running:
            running = self._handle_events()
            self._draw()
            self._clock.tick(60)
        pygame.quit()
