import math
import random
import pygame


class Wanderer:
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
        self.img_x, self.img_y = img_x, img_y
        self.steer_map      = steer_map
        self.brightness_map = brightness_map

        self.marks = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        self.marks.fill((0, 0, 0, 0))

        # Motion
        self.x           = img_w / 2.0
        self.y           = img_h / 2.0
        self.angle       = random.uniform(0, 2 * math.pi)
        self.speed       = 1.0
        self.steps_per_frame = 120

        # Appearance
        self.steer_scale = 10.0
        self.color_index = 0
        self.jitter      = False

        self._last_steer = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_steer(self):
        ix = max(0, min(self.img_w - 1, int(self.x)))
        iy = max(0, min(self.img_h - 1, int(self.y)))
        t  = self.steer_map[iy, ix]
        return self.steer_scale * math.tan(t * math.pi / 2) / 10

    def _draw_mark(self):
        c  = (0, 0, 0, 255)
        sx = int(self.x) + self.img_x
        sy = int(self.y) + self.img_y
        if self.jitter:
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
        for _ in range(self.steps_per_frame):
            steer = self._current_steer()
            self._last_steer = steer
            if steer > 0:
                self.angle += random.gauss(0, steer)

            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed

            margin = 2
            if self.x < margin:
                self.x = margin;              self.angle = math.pi - self.angle
            elif self.x > self.img_w - margin:
                self.x = self.img_w - margin; self.angle = math.pi - self.angle
            if self.y < margin:
                self.y = margin;              self.angle = -self.angle
            elif self.y > self.img_h - margin:
                self.y = self.img_h - margin; self.angle = -self.angle

            self._draw_mark()

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
        return (int(self.x) + self.img_x, int(self.y) + self.img_y)

    def clear(self):
        self.marks.fill((0, 0, 0, 0))
