import pygame
import sys

from pipeline import build_steer_map, pil_to_surface
from algorithms import Wanderer

WIN_W, WIN_H = 1800, 1400
PAD = 30       # top strip for HUD

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

if len(sys.argv) >= 2:
    image_path = sys.argv[1]
else:
    try:
        with open('/tmp/pencil_last_image.txt') as f:
            image_path = f.read().strip()
        if not image_path:
            raise ValueError
    except Exception:
        print("Usage: python3 main.py <image_path>")
        sys.exit(1)

with open('/tmp/pencil_last_image.txt', 'w') as f:
    f.write(image_path)

pygame.init()

pil_img, steer_map, brightness_map = build_steer_map(image_path, WIN_W, WIN_H - PAD)
IMG_W, IMG_H = pil_img.size
IMG_X = (WIN_W - IMG_W) // 2
IMG_Y = PAD + (WIN_H - PAD - IMG_H) // 2

bw_surface = pil_to_surface(pil_img)

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Pencil Marker")
clock  = pygame.time.Clock()
FONT   = pygame.font.SysFont("monospace", 14)

# ---------------------------------------------------------------------------
# Algorithm registry  —  add new ones here
# ---------------------------------------------------------------------------

algorithms = [
    Wanderer(WIN_W, WIN_H, IMG_W, IMG_H, IMG_X, IMG_Y, steer_map, brightness_map),
]
algo_index = 0
algo = algorithms[algo_index]

show_bg = True

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                algo.clear()
            elif event.key == pygame.K_b:
                show_bg = not show_bg
            elif event.key == pygame.K_TAB:
                algo_index = (algo_index + 1) % len(algorithms)
                algo = algorithms[algo_index]
            else:
                algo.handle_key(event.key)

    algo.update()

    # Composite
    screen.fill((30, 30, 30))
    if show_bg:
        screen.blit(bw_surface, (IMG_X, IMG_Y))
    else:
        pygame.draw.rect(screen, (245, 240, 230), (IMG_X, IMG_Y, IMG_W, IMG_H))
    screen.blit(algo.marks, (0, 0))

    # Dot
    pygame.draw.circle(screen, (220, 60, 60), algo.dot_pos(), 5, 2)

    # HUD
    bg_str  = "ON" if show_bg else "OFF"
    hud_txt = (f"[{algo.name}]  {algo.hud_text()}"
               f"  SPACE=clear  B=bg({bg_str})  TAB=switch  ESC=quit")
    hud = FONT.render(hud_txt, True, (200, 200, 200))
    screen.blit(hud, (10, (PAD - hud.get_height()) // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
