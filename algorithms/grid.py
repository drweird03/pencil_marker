import random
import numpy as np
import pygame
from scipy.spatial import KDTree


class Grid:
    """
    A stippling algorithm that renders an image as a single continuous curve.

    Pipeline overview:
      1. GRID BUILDING  — divide the image into equal-sized cells and measure the
                          mean brightness of each cell from the blurred source image.
      2. DOT ASSIGNMENT — assign each cell a dot count proportional to its darkness
                          using a cubic curve: dot_count = max_dots * (1 - brightness)³.
                          The cubic exponent creates strong contrast — very bright
                          cells get almost no dots while dark cells get many more.
      3. PATH PLANNING  — order the non-empty cells using a nearest-neighbor TSP
                          heuristic (via KD-tree) biased toward darker cells, so the
                          path naturally gravitates toward ink-heavy regions first.
      4. DRAWING        — animate the path cell by cell, placing dots uniformly at
                          random within each cell and connecting every dot to the
                          previous with a smooth Catmull-Rom spline. The result is
                          one long continuous curve threaded through the whole image.

    Controls (when this algorithm is active):
        [  /  ]     decrease / increase cell size (rebuilds everything)
        -  /  =     decrease / increase max dots per darkest cell (rebuilds)
        ,  /  .     decrease / increase dot-count preference in path planning
        ↑  /  ↓     decrease / increase drawing speed (cells per frame)
        G           toggle grid line overlay
        SPACE       clear and restart
    """

    name = "Grid"

    def __init__(self, win_w, win_h, img_w, img_h, img_x, img_y, steer_map, brightness_map):
        self.win_w, self.win_h = win_w, win_h
        self.img_w, self.img_h = img_w, img_h
        self.img_x, self.img_y = img_x, img_y  # pixel offset of image within window
        self.steer_map      = steer_map
        self.brightness_map = brightness_map   # float32 [h, w], 0=dark → 1=bright

        # marks spans the full window so it composites cleanly over any background.
        self.marks      = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        self.marks.fill((0, 0, 0, 0))
        # grid_lines is a separate overlay so toggling it never touches the drawing.
        self.grid_lines = pygame.Surface((win_w, win_h), pygame.SRCALPHA)

        # All dot positions in the order they were drawn. Used to connect dots
        # with a smooth spline — the curve treats this as one long flat sequence
        # with no awareness of cell boundaries.
        self._all_dots = []

        # --- Tunable parameters ---
        self.cell_size       = 10    # side length of each grid cell in pixels
        self.max_dots        = 20    # dot count for the darkest possible cell
        self.dot_preference  = 2.0   # path-planner bias: higher = prefer darker cells
        self.show_grid       = False
        self.cells_per_frame = 8     # how many cells to draw per frame (animation speed)

        # --- Internal state ---
        self._cells    = []   # list of (x0, y0, x1, y1, dot_count) for every cell
        self._path     = []   # cell indices in planned visit order
        self._path_pos = 0    # index into _path — how far the animation has progressed
        self._cur_cell = None # the cell currently being drawn (for dot_pos HUD)

        self._build_grid()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self):
        """
        Partition the image into a uniform grid and compute each cell's dot count.

        dot_count = int(max_dots * (1 - brightness)³)

        The cubic exponent makes the mapping highly nonlinear: a cell at 50%
        brightness gets only 12.5% of max_dots, while a cell at 90% darkness
        gets 72.9%. This produces strong separation between light and dark areas.
        """
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
                # Average the blurred brightness map over this cell
                brightness = self.brightness_map[y0:y1, x0:x1].mean()
                dot_count  = int(self.max_dots * (1.0 - brightness) ** 3)
                cells.append((x0, y0, x1, y1, dot_count))

        self._cells = cells
        self._render_grid_lines(cols, rows, cs)
        self._build_path()

    def _build_path(self):
        """
        Order the non-empty cells using nearest-neighbor TSP with a KD-tree.

        Standard nearest-neighbor always picks the geometrically closest unvisited
        cell. Here we add a dot-count bias by dividing the distance by a factor
        that grows with the cell's darkness:

            score = distance / (1 + dot_preference × normalized_dot_count)

        A lower score is preferred. At dot_preference=0 this reduces to pure
        distance (standard NN). Higher values cause the path to favour darker
        cells even when they are further away, clustering ink-heavy regions early.

        The KD-tree makes each nearest-neighbor query O(log n) instead of O(n),
        keeping the total precomputation time fast even for thousands of cells.
        When all k=20 nearest neighbors are already visited, we fall back to a
        global linear search over all remaining unvisited cells.
        """
        # Only visit cells that will actually receive dots
        active_idxs = [i for i, c in enumerate(self._cells) if c[4] > 0]
        if not active_idxs:
            self._path     = []
            self._path_pos = 0
            return

        cells = self._cells

        # Cell centers in image-space (used for distance calculations)
        centers = np.array([
            ((cells[i][0] + cells[i][2]) / 2.0,
             (cells[i][1] + cells[i][3]) / 2.0)
            for i in active_idxs
        ])

        # Normalize dot counts to [0, 1] so the bias is scale-independent
        dot_counts = np.array([cells[i][4] for i in active_idxs], dtype=float)
        norm_dots  = dot_counts / (dot_counts.max() or 1.0)

        n    = len(active_idxs)
        k    = min(20, n)          # number of neighbors to inspect each step
        tree = KDTree(centers)     # spatial index for fast nearest-neighbor lookup

        visited    = np.zeros(n, dtype=bool)
        local_path = []            # indices into active_idxs, filled in visit order
        cur        = 0             # start at the first active cell
        visited[0] = True
        local_path.append(0)

        for _ in range(n - 1):
            dists, neighbors = tree.query(centers[cur], k=k)

            best       = None
            best_score = float('inf')
            for dist, nb in zip(dists, neighbors):
                if visited[nb]:
                    continue
                # Bias score: cells with more dots are treated as "closer"
                score = dist / (1.0 + self.dot_preference * norm_dots[nb])
                if score < best_score:
                    best_score = score
                    best = nb

            if best is None:
                # All k nearest are visited — scan every remaining cell globally
                unvisited = np.where(~visited)[0]
                d = np.linalg.norm(centers[unvisited] - centers[cur], axis=1)
                s = d / (1.0 + self.dot_preference * norm_dots[unvisited])
                best = unvisited[np.argmin(s)]

            visited[best] = True
            local_path.append(best)
            cur = best

        # Map local indices back to indices into self._cells
        self._path     = [active_idxs[i] for i in local_path]
        self._path_pos = 0
        self._cur_cell = None

    def _render_grid_lines(self, cols, rows, cs):
        """
        Pre-render the grid line overlay onto a dedicated surface.
        Drawn once per grid rebuild; toggled at no cost each frame.
        """
        self.grid_lines.fill((0, 0, 0, 0))
        color = (80, 80, 200, 80)  # semi-transparent blue
        for col in range(cols + 1):
            x = col * cs + self.img_x
            pygame.draw.line(self.grid_lines, color,
                             (x, self.img_y), (x, self.img_y + self.img_h))
        for row in range(rows + 1):
            y = row * cs + self.img_y
            pygame.draw.line(self.grid_lines, color,
                             (self.img_x, y), (self.img_x + self.img_w, y))

    def _catmull_rom(self, p0, p1, p2, p3, n=10):
        """
        Return n+1 points along the cubic Bézier segment from p1 to p2 using
        Catmull-Rom parameterization.

        Catmull-Rom automatically derives smooth control points from the four
        surrounding waypoints so the curve passes through p1 and p2 and is
        tangent to the chord p0→p2 at p1 and p1→p3 at p2:

            cp1 = p1 + (p2 - p0) / 6
            cp2 = p2 - (p3 - p1) / 6

        The resulting cubic Bézier is then evaluated at n evenly spaced t values
        and returned as integer pixel coordinates for pygame.draw.lines.
        """
        cp1 = (p1[0] + (p2[0] - p0[0]) / 6.0,
               p1[1] + (p2[1] - p0[1]) / 6.0)
        cp2 = (p2[0] - (p3[0] - p1[0]) / 6.0,
               p2[1] - (p3[1] - p1[1]) / 6.0)
        pts = []
        for i in range(n + 1):
            t  = i / n
            mt = 1 - t
            x  = mt**3*p1[0] + 3*mt**2*t*cp1[0] + 3*mt*t**2*cp2[0] + t**3*p2[0]
            y  = mt**3*p1[1] + 3*mt**2*t*cp1[1] + 3*mt*t**2*cp2[1] + t**3*p2[1]
            pts.append((int(x), int(y)))
        return pts

    def _draw_cell(self, x0, y0, x1, y1, dot_count):
        """
        Place dot_count dots at random positions within the cell and connect
        each new dot to the previous one via a Catmull-Rom spline segment.

        Every dot is appended to self._all_dots. Once we have at least 4 dots,
        we use the most recent 4 as the Catmull-Rom window (p0, p1, p2, p3) and
        draw the smooth segment from p1 to p2. For the very first pair of dots
        (before 4 are available) we fall back to a plain straight line.

        Because _all_dots is flat with no cell-boundary markers, the spline
        flows continuously across cell transitions — the curve has no concept
        of where one cell ends and the next begins.
        """
        for _ in range(dot_count):
            px = random.randint(x0, x1 - 1) + self.img_x
            py = random.randint(y0, y1 - 1) + self.img_y
            self._all_dots.append((px, py))
            n = len(self._all_dots)
            if n >= 4:
                # Draw smooth spline segment through the last 4 dots
                pts = self._catmull_rom(*self._all_dots[-4:])
                pygame.draw.lines(self.marks, (0, 0, 0, 255), False, pts, 1)
            elif n == 2:
                # Not enough points for a spline yet — straight line as fallback
                pygame.draw.line(self.marks, (0, 0, 0, 255),
                                 self._all_dots[-2], self._all_dots[-1], 1)

    # ------------------------------------------------------------------
    # Public interface (called by main.py)
    # ------------------------------------------------------------------

    def update(self):
        """
        Advance the drawing animation by cells_per_frame cells along the path.

        Each call pops the next cell from _path, draws its dots (which extends
        _all_dots and draws the spline segments), then moves on. When _path_pos
        reaches the end of _path the drawing is complete and update() becomes a no-op.
        """
        for _ in range(self.cells_per_frame):
            if self._path_pos >= len(self._path):
                break
            idx  = self._path[self._path_pos]
            cell = self._cells[idx]
            self._cur_cell = cell
            self._draw_cell(*cell)
            self._path_pos += 1

    def handle_key(self, key):
        if key == pygame.K_LEFTBRACKET:
            self.cell_size = max(4, self.cell_size - 2)
            self._build_grid();                                       return True
        elif key == pygame.K_RIGHTBRACKET:
            self.cell_size += 2
            self._build_grid();                                       return True
        elif key == pygame.K_MINUS:
            self.max_dots = max(1, self.max_dots - 5)
            self._build_grid();                                       return True
        elif key == pygame.K_EQUALS:
            self.max_dots += 5
            self._build_grid();                                       return True
        elif key == pygame.K_COMMA:
            self.dot_preference = max(0.0, round(self.dot_preference - 0.5, 1))
            self._build_path();                                       return True
        elif key == pygame.K_PERIOD:
            self.dot_preference = round(self.dot_preference + 0.5, 1)
            self._build_path();                                       return True
        elif key == pygame.K_UP:
            self.cells_per_frame = min(200, self.cells_per_frame + 4); return True
        elif key == pygame.K_DOWN:
            self.cells_per_frame = max(1,   self.cells_per_frame - 4); return True
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid;                      return True
        return False

    def hud_text(self):
        done  = self._path_pos
        total = len(self._path)
        g = "ON" if self.show_grid else "OFF"
        return (f"[/]=cell({self.cell_size}px)  -/+=dots({self.max_dots})"
                f"  ,/.=pref({self.dot_preference})  up/dn=spd({self.cells_per_frame})"
                f"  G=grid({g})  {done}/{total} cells")

    def dot_pos(self):
        """Return the center of the current cell in window coordinates for the HUD dot."""
        if self._cur_cell:
            x0, y0, x1, y1, _ = self._cur_cell
            return ((x0 + x1) // 2 + self.img_x,
                    (y0 + y1) // 2 + self.img_y)
        return (self.img_x + self.img_w // 2, self.img_y + self.img_h // 2)

    def clear(self):
        self.marks.fill((0, 0, 0, 0))
        self._all_dots = []
        self._build_grid()
