"""
Settings panel — draggable sliders drawn inside a single pygame surface.

Everything is rendered in panel-local coordinates, then blitted to the screen
as one unit. This avoids any mismatch between the background surface and the
widgets drawn on top of it.

Usage (main.py):
    panel = SettingsPanel(WIN_W, WIN_H)
    panel.set_algo(algo)

    # inside event loop (before key handling):
    if panel.handle_event(event):
        continue

    # inside render loop (after everything else):
    panel.draw(screen)

Each algorithm opts in by implementing get_sliders(), which returns a list of:
    dict(label=str, attr=str, min=num, max=num, step=num, rebuild=str|None)
"""

import math
import pygame

# ── colours ────────────────────────────────────────────────────────────────
_BG           = (22,  22,  22)
_BORDER       = (55,  55,  55)
_TITLE_FG     = (235, 235, 235)
_LABEL_FG     = (170, 170, 170)
_VALUE_FG     = (100, 205, 100)
_TRACK_BG     = (60,  60,  60)
_TRACK_FILL   = (65,  130, 240)
_HANDLE_IDLE  = (215, 215, 215)
_HANDLE_DRAG  = (65,  130, 240)
_HINT_FG      = (70,  70,  70)
_PENDING_FG   = (240, 165,  40)
_TAB_BG       = (40,  40,  40)
_TAB_FG       = (160, 160, 160)

# heavier rebuild > lighter rebuild  (None = no rebuild needed)
_REBUILD_COST = {'_build_grid': 2, '_build_path': 1}


# ── slider ─────────────────────────────────────────────────────────────────

class _Slider:
    """
    One labelled slider.  All coordinates are panel-local (origin = top-left
    corner of the panel surface), so the slider can be drawn onto the panel
    surface directly without any offset arithmetic.
    """

    TRACK_H  = 6
    HANDLE_R = 10

    def __init__(self, lx, ty, width, label, val_min, val_max, value, step,
                 attr, rebuild):
        self.lx      = lx         # left x of track (panel-local)
        self.ty      = ty         # top  y of track  (panel-local)
        self.width   = width
        self.label   = label
        self.min     = float(val_min)
        self.max     = float(val_max)
        self.value   = float(value)
        self.step    = float(step)
        self.attr    = attr
        self.rebuild = rebuild
        self.dragging = False

    # ── helpers ──────────────────────────────────────────────────────────

    @property
    def _cy(self):
        return self.ty + 10           # centre-y of track

    def _val_to_x(self, val):
        t = (val - self.min) / (self.max - self.min)
        return self.lx + int(t * self.width)

    def _x_to_val(self, local_x):
        t   = max(0.0, min(1.0, (local_x - self.lx) / self.width))
        raw = self.min + t * (self.max - self.min)
        return max(self.min, min(self.max,
                                 round(raw / self.step) * self.step))

    def _fmt(self, v):
        if self.step >= 1:
            return str(int(round(v)))
        decs = max(0, -int(math.floor(math.log10(self.step))))
        return f"{v:.{decs}f}"

    # ── interaction ──────────────────────────────────────────────────────

    def hit_handle(self, local_pos):
        hx = self._val_to_x(self.value)
        return math.hypot(local_pos[0] - hx, local_pos[1] - self._cy) \
               <= self.HANDLE_R + 6

    def drag_to(self, local_x):
        """Update value from a drag. Returns True if value changed."""
        new = self._x_to_val(local_x)
        if new != self.value:
            self.value = new
            return True
        return False

    # ── drawing ──────────────────────────────────────────────────────────

    def draw(self, surf, font, pending=False):
        cy  = self._cy
        lx  = self.lx
        rx  = lx + self.width
        hx  = self._val_to_x(self.value)

        # label (left) and value (right), one row above the track
        label_s = font.render(self.label, True, _LABEL_FG)
        value_col = _PENDING_FG if pending else _VALUE_FG
        value_s = font.render(self._fmt(self.value), True, value_col)
        row_top = self.ty - 20
        surf.blit(label_s, (lx, row_top))
        surf.blit(value_s, (rx - value_s.get_width(), row_top))

        # track background
        th = self.TRACK_H
        tr = pygame.Rect(lx, cy - th // 2, self.width, th)
        pygame.draw.rect(surf, _TRACK_BG, tr, border_radius=3)

        # filled portion
        if hx > lx:
            fr = pygame.Rect(lx, tr.y, hx - lx, th)
            pygame.draw.rect(surf, _TRACK_FILL, fr, border_radius=3)

        # handle
        fill = _HANDLE_DRAG if self.dragging else _HANDLE_IDLE
        pygame.draw.circle(surf, fill, (hx, cy), self.HANDLE_R)
        pygame.draw.circle(surf, _TRACK_FILL if self.dragging else _TRACK_BG,
                           (hx, cy), self.HANDLE_R, 2)


# ── panel ──────────────────────────────────────────────────────────────────

class SettingsPanel:
    """
    Right-side settings panel.  Press U to show/hide.

    The entire panel is rendered into a dedicated surface each frame; that
    surface is then blitted onto the screen at (win_w - W, 0).  Slider
    coordinates are panel-local, so there is no possibility of an offset bug.
    """

    W      = 270    # panel width in pixels
    PAD    = 18     # horizontal padding inside panel
    ROW_H  = 64     # pixels per slider row
    TOP    = 72     # y of first slider row (panel-local)
    TAB_W  = 28     # width of the "U" tab shown when panel is hidden

    def __init__(self, win_w, win_h):
        self.win_w    = win_w
        self.win_h    = win_h
        self.visible  = False
        self.algo     = None
        self.sliders: list[_Slider] = []
        self._dirty   = None     # rebuild method to call on mouse release
        self._surf    = None     # panel surface, recreated when needed
        self._font    = None     # initialised on first draw()

    # ── public ───────────────────────────────────────────────────────────

    def set_algo(self, algo):
        self.algo    = algo
        self._dirty  = None
        self._build_sliders()

    def toggle(self):
        self.visible = not self.visible

    def handle_event(self, event) -> bool:
        """
        Returns True if the panel consumed the event.
        """
        if not self.visible:
            return False

        panel_x = self.win_w - self.W

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[0] < panel_x:
                return False                     # click left of panel
            local = (event.pos[0] - panel_x, event.pos[1])
            for s in self.sliders:
                if s.hit_handle(local):
                    s.dragging = True
                    self._dirty = None
                    return True
            return True                          # inside panel but no handle hit

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_dragging = any(s.dragging for s in self.sliders)
            for s in self.sliders:
                s.dragging = False
            if was_dragging and self._dirty and self.algo:
                getattr(self.algo, self._dirty)()
                self._dirty = None
                self._sync()
            return was_dragging

        elif event.type == pygame.MOUSEMOTION:
            for s in self.sliders:
                if s.dragging:
                    local_x = event.pos[0] - (self.win_w - self.W)
                    if s.drag_to(local_x):
                        orig = getattr(self.algo, s.attr)
                        setattr(self.algo, s.attr, type(orig)(s.value))
                        cost = _REBUILD_COST.get(s.rebuild, 0)
                        if cost > _REBUILD_COST.get(self._dirty, 0):
                            self._dirty = s.rebuild
                    return True

        return False

    def draw(self, screen):
        # always draw the small tab so the user knows U exists
        self._ensure_font()
        if not self.visible or self.algo is None:
            self._draw_tab(screen)
            return

        self._sync()

        # build the panel surface
        surf = pygame.Surface((self.W, self.win_h))
        surf.fill(_BG)

        # left border
        pygame.draw.line(surf, _BORDER, (0, 0), (0, self.win_h - 1))

        # title
        title = self._font.render(f"{self.algo.name}  —  settings", True, _TITLE_FG)
        surf.blit(title, (self.PAD, 26))
        pygame.draw.line(surf, _BORDER, (0, 52), (self.W, 52))

        # sliders
        for s in self.sliders:
            s.draw(surf, self._font,
                   pending=(self._dirty is not None and s.dragging))

        # footer hints
        lines = []
        if self._dirty:
            lines.append(("↑ release to apply", _PENDING_FG))
        lines.append(("U  hide  |  R  reload config", _HINT_FG))
        for i, (text, col) in enumerate(lines):
            hs = self._font.render(text, True, col)
            surf.blit(hs, (self.PAD, self.win_h - 46 + i * 20))

        screen.blit(surf, (self.win_w - self.W, 0))

    # ── internal ─────────────────────────────────────────────────────────

    def _ensure_font(self):
        if self._font is None:
            self._font = (pygame.font.SysFont("monospace", 13)
                          or pygame.font.Font(None, 16))

    def _draw_tab(self, screen):
        """Small 'U' tab on the right edge when panel is hidden."""
        tx = self.win_w - self.TAB_W
        ty = self.win_h // 2 - 30
        tab = pygame.Rect(tx, ty, self.TAB_W, 60)
        pygame.draw.rect(screen, _TAB_BG, tab, border_radius=4)
        pygame.draw.rect(screen, _BORDER,  tab, 1, border_radius=4)
        label = self._font.render("U", True, _TAB_FG)
        lx = tx + (self.TAB_W - label.get_width()) // 2
        ly = ty + (60 - label.get_height()) // 2
        screen.blit(label, (lx, ly))

    def _build_sliders(self):
        if self.algo is None or not hasattr(self.algo, 'get_sliders'):
            self.sliders = []
            return
        specs = self.algo.get_sliders()
        lx    = self.PAD
        tw    = self.W - self.PAD * 2
        self.sliders = []
        for i, sp in enumerate(specs):
            ty  = self.TOP + i * self.ROW_H + 26   # track top (panel-local)
            val = getattr(self.algo, sp['attr'])
            self.sliders.append(
                _Slider(lx, ty, tw, sp['label'],
                        sp['min'], sp['max'], val, sp['step'],
                        sp['attr'], sp['rebuild'])
            )

    def _sync(self):
        """Pull current algo values into sliders (skip ones being dragged)."""
        for s in self.sliders:
            if not s.dragging:
                s.value = float(getattr(self.algo, s.attr))
