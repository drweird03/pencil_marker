import numpy as np
import pygame
from PIL import Image, ImageFilter

# Controls how steeply the steer map rises in dark regions.
# Higher values make bright areas almost perfectly straight (steer ≈ 0)
# while dark areas get an exponentially larger steer value.
GAMMA = 5.0


def build_steer_map(path, max_width, max_height, blur_radius=12):
    """
    Load an image and produce the two maps used by drawing algorithms.

    Steps:
      1. Open and convert to grayscale ('L' mode).
      2. Resize to fit within (max_width, max_height) preserving aspect ratio.
         thumbnail() only ever shrinks, never enlarges.
      3. Apply a Gaussian blur (radius=12 by default) to smooth out noise and
         give each pixel a brightness that represents its broader neighbourhood
         rather than a single pixel value.
      4. Compute brightness as a float in [0, 1] (0=black, 1=white).
      5. Compute the exponential steer map via:
             t        = 1 - brightness          (invert: dark → high t)
             exp_map  = (exp(GAMMA * t) - 1) / (exp(GAMMA) - 1)
         This maps t through an exponential curve normalised to [0, 1].
         Bright areas → t ≈ 0 → exp_map ≈ 0 (almost no steer).
         Dark areas   → t ≈ 1 → exp_map ≈ 1 (maximum steer).

    Returns:
        img        — PIL Image (grayscale, resized) used for display.
        exp_map    — float32 numpy array [h, w] in [0, 1]; used by Wanderer
                     to drive chaotic motion in dark image areas.
        brightness — float32 numpy array [h, w] in [0, 1]; used by Grid to
                     determine dot density, and by Wanderer to check ink coverage.
    """
    img = Image.open(path).convert('L')
    img.thumbnail((max_width, max_height), Image.LANCZOS)

    blurred    = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    brightness = np.array(blurred, dtype=float) / 255.0  # 0=dark, 1=bright

    t       = 1.0 - brightness
    exp_map = (np.exp(GAMMA * t) - 1) / (np.exp(GAMMA) - 1)  # 0=bright, 1=dark

    return img, exp_map, brightness


def pil_to_surface(pil_img):
    """Convert a PIL Image to a pygame Surface for direct blitting."""
    return pygame.image.fromstring(pil_img.convert('RGB').tobytes(), pil_img.size, 'RGB')
