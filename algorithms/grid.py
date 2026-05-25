import importlib
import math
import random
import numpy as np
import pygame
from PIL import Image as PILImage, ImageDraw
from scipy.spatial import KDTree

import config

# Path where the vector SVG is saved after each render
SVG_OUTPUT_PATH = '/tmp/pencil_marker.svg'


class Grid:
    """
    A stippling algorithm that renders an image as a single continuous spline.

    Pipeline:
      1. GRID BUILDING  — partition image into cells, measure brightness per cell.
      2. DOT ASSIGNMENT — power curve: dot_count = max_dots * (1 - brightness)^dot_power
      3. PATH PLANNING  — nearest-neighbor TSP (KD-tree) biased toward dark cells.
      4. DOT GENERATION — pre-calculate all dot (x, y) positions, one list per cell.
      5. CLUSTER MERGE  — consecutive same-length cell arrays are merged + shuffled,
                          interleaving dots of equal density across adjacent cells.
      6. FLATTEN        — one ordered flat list of all dot positions.
      7. SVG EXPORT     — convert the dot sequence into a vector SVG file using
                          Catmull-Rom cubic Bézier path commands. Saved to disk.
      8. RASTERIZE      — render the spline to a PIL RGBA image using the same
                          Catmull-Rom math, then load as a pygame surface (marks).
                          No pygame.draw calls — the image IS the marks layer.

    Tunable parameters (edit config.py then press R to reload):
      CELL_SIZE       grid cell size in pixels
      MAX_DOTS        max dots per cell at full darkness
      DOT_POWER       exponent for dot count curve (1=linear, 2=quadratic, 3=cubic)
      DOT_PREFERENCE  TSP bias toward darker cells
      SPLINE_N        Catmull-Rom interpolation steps per segment

    Keyboard shortcuts (runtime fine-tuning):
      [ / ]    cell_size ±2
      - / =    max_dots ±5
      O / P    dot_power ±1
      , / .    dot_preference ±0.5
      G        toggle grid overlay
      R        reload config.py and rebuild
    """

    name = "Grid"

    def __init__(self, win_w, win_h, img_w, img_h, img_x, img_y, steer_map, brightness_map):
        self.win_w, self.win_h = win_w, win_h
        self.img_w, self.img_h = img_w, img_h
        self.img_x, self.img_y = img_x, img_y
        self.steer_map      = steer_map
        self.brightness_map = brightness_map

        self.marks      = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        self.marks.fill((0, 0, 0, 0))
        self.grid_lines = pygame.Surface((win_w, win_h), pygame.SRCALPHA)

        # Load initial values from config
        self.cell_size      = config.CELL_SIZE
        self.max_dots       = config.MAX_DOTS
        self.dot_power      = config.DOT_POWER
        self.dot_preference = config.DOT_PREFERENCE
        self.spline_n       = config.SPLINE_N
        self.show_grid      = False

        self._cells        = []
        self._path         = []
        self._all_dots     = []
        self._curve_length = 0.0

        self._build_grid()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self):
        cs   = self.cell_size
        cols = max(1, self.img_w // cs)
        rows = max(1, self.img_h // cs)

        cells = []
        for row in range(rows):
            for col in range(cols):
                x0 = col * cs
                y0 = row * cs
                x1 = min(x0 + cs, self.img_w)
                y1 = min(y0 + cs, self.img_h)
                brightness = self.brightness_map[y0:y1, x0:x1].mean()
                dot_count  = int(self.max_dots * (1.0 - brightness) ** self.dot_power)
                cells.append((x0, y0, x1, y1, dot_count))

        self._cells = cells
        self._render_grid_lines(cols, rows, cs)
        self._build_path()

    def _build_path(self):
        """Build the ordered dot sequence, export SVG, and rasterize to marks."""
        active_idxs = [i for i, c in enumerate(self._cells) if c[4] > 0]
        if not active_idxs:
            self._path     = []
            self._all_dots = []
            self.marks.fill((0, 0, 0, 0))
            return

        cells = self._cells

        # Nearest-neighbor TSP with KD-tree, biased toward darker cells
        centers = np.array([
            ((cells[i][0] + cells[i][2]) / 2.0,
             (cells[i][1] + cells[i][3]) / 2.0)
            for i in active_idxs
        ])
        dot_counts = np.array([cells[i][4] for i in active_idxs], dtype=float)
        norm_dots  = dot_counts / (dot_counts.max() or 1.0)

        n    = len(active_idxs)
        k    = min(20, n)
        tree = KDTree(centers)

        visited    = np.zeros(n, dtype=bool)
        local_path = [0]
        visited[0] = True
        cur        = 0

        for _ in range(n - 1):
            dists, neighbors = tree.query(centers[cur], k=k)
            best = None;  best_score = float('inf')
            for dist, nb in zip(dists, neighbors):
                if visited[nb]:
                    continue
                # Prefer neighbors whose dot density is similar to the current cell
                similarity = 1.0 - abs(norm_dots[nb] - norm_dots[cur])
                score = dist / (1.0 + self.dot_preference * similarity)
                if score < best_score:
                    best_score = score;  best = nb
            if best is None:
                unvisited = np.where(~visited)[0]
                d = np.linalg.norm(centers[unvisited] - centers[cur], axis=1)
                similarity = 1.0 - np.abs(norm_dots[unvisited] - norm_dots[cur])
                s = d / (1.0 + self.dot_preference * similarity)
                best = unvisited[np.argmin(s)]
            visited[best] = True
            local_path.append(best)
            cur = best

        self._path = [active_idxs[i] for i in local_path]

        # Pre-calculate dot positions per cell — Gaussian around cell centre,
        # σ = cell_size so dots can bleed softly into neighbouring cells.
        cell_dots = []
        for idx in self._path:
            x0, y0, x1, y1, dot_count = cells[idx]
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            sd = self.cell_size
            dots = []
            for _ in range(dot_count):
                dx = max(0, min(self.img_w - 1, random.gauss(cx, sd)))
                dy = max(0, min(self.img_h - 1, random.gauss(cy, sd)))
                dots.append((int(dx) + self.img_x, int(dy) + self.img_y))
            cell_dots.append(dots)

        # Cluster merge: consecutive same-length arrays where every new cell is
        # 8-connected (shares edge or corner) to at least one cell already in
        # the cluster → merge + shuffle.
        cols = max(1, self.img_w // self.cell_size)

        def adjacent_to_cluster(idx, cluster_idxs):
            r, c = idx // cols, idx % cols
            for ci in cluster_idxs:
                cr, cc = ci // cols, ci % cols
                if abs(r - cr) <= 1 and abs(c - cc) <= 1:
                    return True
            return False

        merged = []
        i = 0
        while i < len(cell_dots):
            length       = len(cell_dots[i])
            cluster      = list(cell_dots[i])
            cluster_idxs = [self._path[i]]
            j = i + 1
            while j < len(cell_dots):
                nxt = self._path[j]
                if len(cell_dots[j]) == length and adjacent_to_cluster(nxt, cluster_idxs):
                    cluster.extend(cell_dots[j])
                    cluster_idxs.append(nxt)
                    j += 1
                else:
                    break
            random.shuffle(cluster)
            merged.append(cluster)
            i = j

        # Flatten into one ordered sequence
        self._all_dots = [dot for group in merged for dot in group]

        # Export vector SVG then rasterize to marks
        svg = self._build_svg()
        with open(SVG_OUTPUT_PATH, 'w') as f:
            f.write(svg)
        self._rasterize()

    def _catmull_rom_points(self, p0, p1, p2, p3, n=None):
        """
        Return n+1 interpolated points along the Catmull-Rom segment p1 → p2.
        Uses self.spline_n by default. Used both for SVG control-point math
        and PIL rasterization.
        """
        if n is None:
            n = self.spline_n
        cp1 = (p1[0] + (p2[0] - p0[0]) / 6.0,
               p1[1] + (p2[1] - p0[1]) / 6.0)
        cp2 = (p2[0] - (p3[0] - p1[0]) / 6.0,
               p2[1] - (p3[1] - p1[1]) / 6.0)
        pts = []
        for i in range(n + 1):
            t  = i / n;  mt = 1 - t
            x  = mt**3*p1[0] + 3*mt**2*t*cp1[0] + 3*mt*t**2*cp2[0] + t**3*p2[0]
            y  = mt**3*p1[1] + 3*mt**2*t*cp1[1] + 3*mt*t**2*cp2[1] + t**3*p2[1]
            pts.append((x, y))
        return pts, cp1, cp2

    def _build_svg(self):
        """
        Build an SVG string representing the full Catmull-Rom spline.
        Each segment p1→p2 becomes a cubic Bézier C command, with control
        points derived from the Catmull-Rom formula. The result is an exact
        vector representation of the curve — no pixel approximation.
        """
        dots = self._all_dots
        if len(dots) < 2:
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.win_w}" height="{self.win_h}"/>'

        parts = [f'M {dots[0][0]},{dots[0][1]} L {dots[1][0]},{dots[1][1]}']

        for i in range(3, len(dots)):
            p0, p1, p2, p3 = dots[i-3], dots[i-2], dots[i-1], dots[i]
            _, cp1, cp2 = self._catmull_rom_points(p0, p1, p2, p3)
            # SVG cubic Bézier: C cp1 cp2 endpoint
            parts.append(
                f'C {cp1[0]:.2f},{cp1[1]:.2f} {cp2[0]:.2f},{cp2[1]:.2f} {p2[0]},{p2[1]}'
            )

        path_d = ' '.join(parts)
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.win_w}" height="{self.win_h}">\n'
            f'  <path d="{path_d}" fill="none" stroke="black" '
            f'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>\n'
            f'</svg>\n'
        )

    def _rasterize(self):
        """
        Render the spline onto a transparent PIL RGBA image using the same
        Catmull-Rom math, then convert to a pygame SRCALPHA surface.
        Replaces self.marks entirely — no pygame.draw calls.
        """
        dots = self._all_dots
        img  = PILImage.new('RGBA', (self.win_w, self.win_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if len(dots) < 2:
            self.marks = pygame.image.fromstring(img.tobytes(), img.size, 'RGBA')
            return

        # Collect all interpolated curve points into one polyline
        all_pts = [dots[0], dots[1]]
        for i in range(3, len(dots)):
            pts, _, _ = self._catmull_rom_points(*dots[i-3:i+1])
            all_pts.extend(pts[1:])  # skip first (already last of previous segment)

        # Measure total arc length
        self._curve_length = sum(
            math.hypot(all_pts[i][0] - all_pts[i-1][0],
                       all_pts[i][1] - all_pts[i-1][1])
            for i in range(1, len(all_pts))
        )

        draw.line(all_pts, fill=(0, 0, 0, 255), width=1)

        raw = img.tobytes()
        self.marks = pygame.image.fromstring(raw, img.size, 'RGBA')

    def _render_grid_lines(self, cols, rows, cs):
        self.grid_lines.fill((0, 0, 0, 0))
        color = (80, 80, 200, 80)
        for col in range(cols + 1):
            x = col * cs + self.img_x
            pygame.draw.line(self.grid_lines, color,
                             (x, self.img_y), (x, self.img_y + self.img_h))
        for row in range(rows + 1):
            y = row * cs + self.img_y
            pygame.draw.line(self.grid_lines, color,
                             (self.img_x, y), (self.img_x + self.img_w, y))

    # ------------------------------------------------------------------
    # Public interface (called by main.py)
    # ------------------------------------------------------------------

    def update(self):
        pass  # all work done upfront in _build_path

    def reload_config(self):
        """Re-read config.py from disk and rebuild the grid."""
        importlib.reload(config)
        self.cell_size      = config.CELL_SIZE
        self.max_dots       = config.MAX_DOTS
        self.dot_power      = config.DOT_POWER
        self.dot_preference = config.DOT_PREFERENCE
        self.spline_n       = config.SPLINE_N
        self._build_grid()

    def handle_key(self, key):
        if key == pygame.K_LEFTBRACKET:
            self.cell_size = max(2, self.cell_size - 2)
            self._build_grid();                                           return True
        elif key == pygame.K_RIGHTBRACKET:
            self.cell_size += 2
            self._build_grid();                                           return True
        elif key == pygame.K_MINUS:
            self.max_dots = max(1, self.max_dots - 5)
            self._build_grid();                                           return True
        elif key == pygame.K_EQUALS:
            self.max_dots += 5
            self._build_grid();                                           return True
        elif key == pygame.K_o:
            self.dot_power = max(1, self.dot_power - 1)
            self._build_grid();                                           return True
        elif key == pygame.K_p:
            self.dot_power += 1
            self._build_grid();                                           return True
        elif key == pygame.K_COMMA:
            self.dot_preference = max(0.0, round(self.dot_preference - 0.5, 1))
            self._build_path();                                           return True
        elif key == pygame.K_PERIOD:
            self.dot_preference = round(self.dot_preference + 0.5, 1)
            self._build_path();                                           return True
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid;                         return True
        return False

    def hud_text(self):
        total = len(self._all_dots)
        g     = "ON" if self.show_grid else "OFF"
        rel   = self._curve_length / max(self.img_w, self.img_h)
        return (f"[/]=cell({self.cell_size})  -/+=dots({self.max_dots})"
                f"  O/P=power({self.dot_power})  ,/.=sim({self.dot_preference})"
                f"  G=grid({g})  {total}pts  len={rel:.1f}x")

    def dot_pos(self):
        if self._all_dots:
            return self._all_dots[-1]
        return (self.img_x + self.img_w // 2, self.img_y + self.img_h // 2)

    def get_sliders(self):
        """Slider definitions consumed by ui.SettingsPanel."""
        return [
            dict(label='Cell Size',  attr='cell_size',      min=2,   max=20,  step=1,   rebuild='_build_grid'),
            dict(label='Max Dots',   attr='max_dots',        min=1,   max=40,  step=1,   rebuild='_build_grid'),
            dict(label='Dot Power',  attr='dot_power',       min=1,   max=8,   step=1,   rebuild='_build_grid'),
            dict(label='Similarity', attr='dot_preference',  min=0.0, max=10.0,step=0.5, rebuild='_build_path'),
            dict(label='Spline N',   attr='spline_n',        min=2,   max=40,  step=1,   rebuild='_build_path'),
        ]

    def clear(self):
        self._build_grid()
