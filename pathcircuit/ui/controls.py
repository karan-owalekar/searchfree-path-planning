from __future__ import annotations

import pygame
from pathcircuit.ui import theme


class Button:
    def __init__(self, x: int, y: int, w: int, h: int, text: str, accent: bool = False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.accent = accent
        self._hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if self.accent:
            color = theme.BTN_ACCENT
        elif self._hovered:
            color = theme.BTN_HOVER
        else:
            color = theme.BTN_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        label = font.render(self.text, True, theme.BTN_TEXT)
        lx = self.rect.x + (self.rect.w - label.get_width()) // 2
        ly = self.rect.y + (self.rect.h - label.get_height()) // 2
        surface.blit(label, (lx, ly))


class Dropdown:
    def __init__(self, x: int, y: int, w: int, h: int, label: str, options: list[str], selected: int = 0):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.options = options
        self.selected = selected
        self._open = False
        self._hovered_option = -1

    @property
    def value(self) -> str:
        return self.options[self.selected]

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if selection changed."""
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._open:
                for i, opt_rect in enumerate(self._option_rects()):
                    if opt_rect.collidepoint(event.pos):
                        if self.selected != i:
                            self.selected = i
                            changed = True
                        self._open = False
                        return changed
                self._open = False
            elif self.rect.collidepoint(event.pos):
                self._open = True
        if event.type == pygame.MOUSEMOTION and self._open:
            self._hovered_option = -1
            for i, opt_rect in enumerate(self._option_rects()):
                if opt_rect.collidepoint(event.pos):
                    self._hovered_option = i
        return changed

    def _option_rects(self) -> list[pygame.Rect]:
        rects = []
        for i in range(len(self.options)):
            rects.append(pygame.Rect(
                self.rect.x,
                self.rect.bottom + i * self.rect.h,
                self.rect.w,
                self.rect.h,
            ))
        return rects

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Label
        lbl = font.render(self.label + ":", True, theme.STATUS_TEXT)
        surface.blit(lbl, (self.rect.x - lbl.get_width() - 6, self.rect.y + (self.rect.h - lbl.get_height()) // 2))

        # Main box
        pygame.draw.rect(surface, theme.DROPDOWN_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, theme.DROPDOWN_BORDER, self.rect, width=1, border_radius=4)
        txt = font.render(self.options[self.selected] + "  \u25bc", True, theme.BTN_TEXT)
        surface.blit(txt, (self.rect.x + 8, self.rect.y + (self.rect.h - txt.get_height()) // 2))

        # Dropdown list
        if self._open:
            for i, opt_rect in enumerate(self._option_rects()):
                bg = theme.DROPDOWN_HOVER if i == self._hovered_option else theme.DROPDOWN_BG
                pygame.draw.rect(surface, bg, opt_rect)
                pygame.draw.rect(surface, theme.DROPDOWN_BORDER, opt_rect, width=1)
                opt_txt = font.render(self.options[i], True, theme.BTN_TEXT)
                surface.blit(opt_txt, (opt_rect.x + 8, opt_rect.y + (opt_rect.h - opt_txt.get_height()) // 2))

    @property
    def is_open(self) -> bool:
        return self._open


class StatusBar:
    def __init__(self, y: int, w: int):
        self.rect = pygame.Rect(0, y, w, theme.STATUS_HEIGHT)
        self.text = "Draw obstacles with mouse  |  Press S / G to set start / goal"

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        pygame.draw.rect(surface, theme.PANEL_BG, self.rect)
        txt = font.render(self.text, True, theme.STATUS_TEXT)
        surface.blit(txt, (12, self.rect.y + (self.rect.h - txt.get_height()) // 2))

    def update_rect(self, y: int, w: int) -> None:
        self.rect = pygame.Rect(0, y, w, theme.STATUS_HEIGHT)


_INFO_LINES: list[tuple[str, str]] = [
    ("Click & drag", "Add / remove obstacles"),
    ("Right-click drag", "Erase obstacles"),
    ("Press S + click", "Set start point (green)"),
    ("Press G + click", "Set goal point (red)"),
    ("Find Path", "Compute & show shortest path"),
    ("Clear Path", "Remove displayed path"),
    ("Clear Grid", "Reset entire grid"),
]


class InfoOverlay:
    """Modal instructions panel toggled by the '?' button."""

    def __init__(self) -> None:
        self.visible = False

    def toggle(self) -> None:
        self.visible = not self.visible

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if the overlay consumed the event."""
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.visible = False
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.visible = False
            return True
        return True

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        font_sm: pygame.font.Font,
        font_title: pygame.font.Font,
    ) -> None:
        if not self.visible:
            return

        sw, sh = surface.get_size()

        backdrop = pygame.Surface((sw, sh), pygame.SRCALPHA)
        backdrop.fill(theme.OVERLAY_BG)
        surface.blit(backdrop, (0, 0))

        line_h = font.get_height() + 6
        pad = 36
        panel_w = 760
        panel_h = pad + font_title.get_height() + 18 + line_h * len(_INFO_LINES) + pad + font_sm.get_height() + 10
        px = (sw - panel_w) // 2
        py = (sh - panel_h) // 2
        panel_rect = pygame.Rect(px, py, panel_w, panel_h)

        pygame.draw.rect(surface, theme.OVERLAY_PANEL, panel_rect, border_radius=14)
        pygame.draw.rect(surface, theme.OVERLAY_BORDER, panel_rect, width=2, border_radius=14)

        heading = font_title.render("How to Use", True, theme.OVERLAY_HEADING)
        surface.blit(heading, (px + (panel_w - heading.get_width()) // 2, py + pad))

        y = py + pad + heading.get_height() + 18
        col2_x = px + pad + 290
        for key, desc in _INFO_LINES:
            k = font.render(key, True, theme.BTN_TEXT)
            d = font_sm.render(desc, True, theme.STATUS_TEXT)
            surface.blit(k, (px + pad, y))
            surface.blit(d, (col2_x, y + (k.get_height() - d.get_height()) // 2))
            y += line_h

        hint = font_sm.render("Click anywhere or press Esc to close", True, theme.STATUS_TEXT)
        surface.blit(hint, (px + (panel_w - hint.get_width()) // 2, y + 8))
