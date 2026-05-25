import importlib
import math
import random
import pygame

import config


class Wanderer:
    """
    A freehand drawing algorithm that simulates a pencil wandering across the image.

    The pencil moves as a random walk, but its randomness (steer) is driven by the
    brightness of the image at its current position. Dark areas produce high steer
    (chaotic, scribbling motion), while bright areas produce low steer (nearly
    straight lines). The result is a single continuous stroke that organically
    fills in the darker regions of the image.

    Ink avoidance: before each step the pencil samples a small patch in its
    intended direction and compares the target image brightness to how much ink
    has already been laid down there. It only steps into spots that still need
    more ink, rotating by π/12 each attempt until a valid direction is found.
    If the best valid direction would send it backward, it continues straight
    instead to avoid doubling back on itself.

    Controls (when this algorithm is active):
        [  /  ]   decrease / increase steer scale
        J         toggle jitter mode (scattered blobs instead of single dots)
        1 – 5     change color preset (currently unused in black mode)
    """

    name = "Wanderer"

    COLORS = [
        (30,  30,  80),
        (20,  80,  20),
        (120, 30,  30),
        (80,  40,   0),
        (40,  40,  40),
    ]

    def __init__(self, win_w, win_h, img_w, img_h, img_x, img_y, steer_map, brightness_map):
        self.win_w, self.win_h = win_w, win_h
        self.img_w, self.img_h = img_w, img_h
        self.img_x, self.img_y = img_x, img_y  # pixel offset of image within window
        self.steer_map      = steer_map      # float32 [h, w], 0=bright → 1=dark (exponential)
        self.brightness_map = brightness_map  # float32 [h, w], 0=dark → 1=bright (linear)

        # The marks surface spans the full window and uses per-pixel alpha so
        # it can be composited over the background without clearing it each frame.
        self.marks = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        self.marks.fill((0, 0, 0, 0))

        # --- Motion state ---
        self.x     = img_w / 2.0                    # current position in image-space
        self.y     = img_h / 2.0
        self.angle = random.uniform(0, 2 * math.pi)  # current travel direction (radians)
        self.speed = 1.0                             # pixels moved per step

        # Number of physics steps computed per frame. More steps = faster coverage
        # without changing the visual frame rate.
        self.steps_per_frame = config.STEPS_PER_FRAME

        # --- Appearance ---
        self.steer_scale = config.STEER_SCALE  # multiplier applied to the raw steer value
        self.color_index = 0      # index into COLORS (currently black only)
        self.jitter      = False  # if True, each mark is a small scattered cluster

        # Diagnostics exposed to the HUD
        self._last_steer      = 0.0
        self._last_move_angle = None  # travel angle of the last successful step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_steer(self):
        """
        Look up the steer value for the current position.

        The steer map stores values in [0, 1] via an exponential curve so that
        bright areas have very low values and dark areas approach 1. We pass
        that through tan(t * π/2) to create an unbounded output — values near
        1 explode toward infinity, giving arbitrarily chaotic motion in the
        darkest regions. The /10 and steer_scale keep the units consistent with
        the Gaussian noise applied to the angle each step.
        """
        ix = max(0, min(self.img_w - 1, int(self.x)))
        iy = max(0, min(self.img_h - 1, int(self.y)))
        t  = self.steer_map[iy, ix]
        return self.steer_scale * math.tan(t * math.pi / 2) / 10

    def _sample_brightness(self, nx, ny, marks_alpha, r=3):
        """
        Sample a small (2r × 2r) patch around candidate position (nx, ny) and
        return (target_brightness, drawn_brightness), both in [0, 1].

        target_brightness  — mean of brightness_map in that patch.
                             1 = bright area of the source image.
        drawn_brightness   — how bright the marks layer is in that patch.
                             1 = unpainted (transparent), 0 = fully inked.
                             Computed as 1 - mean_alpha / 255.

        The two arrays use different coordinate conventions:
          brightness_map  → [row, col]  i.e. [y, x]
          marks_alpha     → [col, row]  i.e. [x, y]  (pygame surfarray is transposed)
        marks_alpha also has an (img_x, img_y) offset because the marks surface
        covers the entire window, not just the image region.
        """
        ix = int(nx)
        iy = int(ny)

        # Clamp to image bounds (brightness_map space)
        by0 = max(0, iy - r);  by1 = min(self.img_h, iy + r)
        bx0 = max(0, ix - r);  bx1 = min(self.img_w, ix + r)
        if bx0 >= bx1 or by0 >= by1:
            return 0.0, 0.0

        target = self.brightness_map[by0:by1, bx0:bx1].mean()

        # Translate to window space and clamp to window bounds (marks_alpha space)
        sx = ix + self.img_x;  sy = iy + self.img_y
        ax0 = max(0, sx - r);  ax1 = min(self.win_w, sx + r)
        ay0 = max(0, sy - r);  ay1 = min(self.win_h, sy + r)
        if ax0 >= ax1 or ay0 >= ay1:
            return target, 0.0

        drawn = 1.0 - marks_alpha[ax0:ax1, ay0:ay1].mean() / 255.0
        return target, drawn

    def _is_backward(self, angle):
        """
        Return True if `angle` points roughly opposite to the last move direction.
        "Roughly" means within one rotation step (π/12) of the exact reverse.
        Used to prevent the pencil from immediately retracing its last stroke.
        """
        if self._last_move_angle is None:
            return False
        back = self._last_move_angle + math.pi
        diff = abs((angle - back + math.pi) % (2 * math.pi) - math.pi)
        return diff < math.pi / 12

    def _draw_mark(self):
        """Draw a single mark at the current position onto the marks surface."""
        c  = (0, 0, 0, 255)
        sx = int(self.x) + self.img_x
        sy = int(self.y) + self.img_y
        if self.jitter:
            # Scatter 6 small circles with Gaussian offsets for a rougher look
            for _ in range(6):
                ox = random.gauss(0, 1.8)
                oy = random.gauss(0, 1.8)
                pygame.draw.circle(self.marks, c,
                                   (sx + int(ox), sy + int(oy)),
                                   random.randint(1, 3))
        else:
            pygame.draw.circle(self.marks, c, (sx, sy), 1)

    # ------------------------------------------------------------------
    # Public interface (called by main.py)
    # ------------------------------------------------------------------

    def update(self):
        """
        Advance the pencil by steps_per_frame steps.

        Each step:
          1. Sample the steer value at the current pixel and add Gaussian noise
             to the travel angle proportional to that steer.
          2. Scan up to 24 candidate directions (stepping π/12 each time) for the
             first direction whose next pixel still needs more ink (target < drawn).
          3. If the best valid direction is backward, continue in the previous
             move direction instead to avoid doubling back.
          4. Move, reflect off image edges if needed, and draw a mark.

        marks_alpha is fetched once per frame (surfarray copy is expensive) and
        shared across all steps.
        """
        marks_alpha = pygame.surfarray.array_alpha(self.marks)  # one copy per frame

        for _ in range(self.steps_per_frame):
            steer = self._current_steer()
            self._last_steer = steer
            if steer > 0:
                self.angle += random.gauss(0, steer)

            # Scan up to 24 angles for the first valid (ink-needed) direction
            first_valid = None
            test_angle  = self.angle
            for _ in range(24):
                nx = self.x + math.cos(test_angle) * self.speed
                ny = self.y + math.sin(test_angle) * self.speed
                target, drawn = self._sample_brightness(nx, ny, marks_alpha)
                if target < drawn:
                    first_valid = (test_angle, nx, ny)
                    break
                test_angle += math.pi / 12

            # Resolve which direction to actually take
            chosen = None
            if first_valid is not None and not self._is_backward(first_valid[0]):
                chosen = first_valid
            elif self._last_move_angle is not None:
                # Best valid direction is backward — keep going straight instead
                nx = self.x + math.cos(self._last_move_angle) * self.speed
                ny = self.y + math.sin(self._last_move_angle) * self.speed
                chosen = (self._last_move_angle, nx, ny)

            if chosen is not None:
                chosen_angle, nx, ny = chosen
                self.angle = chosen_angle
                self.x = nx
                self.y = ny

                # Reflect off image edges like a billiard ball
                margin = 2
                if self.x < margin:
                    self.x = margin;              self.angle = math.pi - self.angle
                elif self.x > self.img_w - margin:
                    self.x = self.img_w - margin; self.angle = math.pi - self.angle
                if self.y < margin:
                    self.y = margin;              self.angle = -self.angle
                elif self.y > self.img_h - margin:
                    self.y = self.img_h - margin; self.angle = -self.angle

                self._last_move_angle = self.angle
                self._draw_mark()
            # else: no valid direction at all — skip this step silently

    def reload_config(self):
        """Re-read config.py from disk and apply new defaults."""
        importlib.reload(config)
        self.steer_scale     = config.STEER_SCALE
        self.steps_per_frame = config.STEPS_PER_FRAME
        self.marks.fill((0, 0, 0, 0))

    def handle_key(self, key):
        if pygame.K_1 <= key <= pygame.K_5:
            self.color_index = key - pygame.K_1;       return True
        elif key == pygame.K_j:
            self.jitter = not self.jitter;             return True
        elif key == pygame.K_LEFTBRACKET:
            self.steer_scale = max(0.0, round(self.steer_scale - 0.05, 4)); return True
        elif key == pygame.K_RIGHTBRACKET:
            self.steer_scale = round(self.steer_scale + 0.05, 4);           return True
        return False

    def hud_text(self):
        j = "ON" if self.jitter else "OFF"
        return (f"1-5=color:{self.color_index+1}  J=jitter({j})"
                f"  [/]=steer({self.steer_scale:.2f})  @dot:{self._last_steer:.3f}")

    def dot_pos(self):
        """Return the current pencil position in window coordinates for the HUD dot."""
        return (int(self.x) + self.img_x, int(self.y) + self.img_y)

    def get_sliders(self):
        """Slider definitions consumed by ui.SettingsPanel."""
        return [
            dict(label='Steer Scale',  attr='steer_scale',     min=0.0, max=30.0, step=0.5, rebuild=None),
            dict(label='Steps/Frame',  attr='steps_per_frame', min=10,  max=500,  step=10,  rebuild=None),
        ]

    def clear(self):
        self.marks.fill((0, 0, 0, 0))
