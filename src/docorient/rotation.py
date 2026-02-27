from __future__ import annotations

from PIL import Image


def apply_rotation(image: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return image.copy()
    return image.rotate(angle, expand=True)
