import numpy as np
import pygame
from PIL import Image, ImageFilter

GAMMA = 5.0  # exponential steepness for steer map

def build_steer_map(path, max_width, max_height, blur_radius=12):
    img = Image.open(path).convert('L')
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    brightness = np.array(blurred, dtype=float) / 255.0
    t = 1.0 - brightness
    exp_map = (np.exp(GAMMA * t) - 1) / (np.exp(GAMMA) - 1)  # [0=bright → 1=dark]
    return img, exp_map, brightness  # brightness: 0=dark, 1=bright

def pil_to_surface(pil_img):
    return pygame.image.fromstring(pil_img.convert('RGB').tobytes(), pil_img.size, 'RGB')
