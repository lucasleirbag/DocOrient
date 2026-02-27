from __future__ import annotations

import numpy as np
from PIL import Image

from docorient._imaging import ImageIO
from docorient.config import OrientationConfig
from docorient.types import OrientationResult

ENERGY_EPSILON = 1e-10


class PrimaryEngine:
    @property
    def name(self) -> str:
        return "primary"

    def detect(self, image: Image.Image, config: OrientationConfig) -> OrientationResult:
        grayscale_image = image.convert("L")
        downscaled_image = ImageIO.downscale(grayscale_image, config.primary_max_dimension)
        if downscaled_image is not grayscale_image:
            grayscale_image.close()

        pixel_array = np.array(downscaled_image, dtype=np.float32)
        downscaled_image.close()
        brightness_threshold = float(pixel_array.mean())

        horizontal_energy, vertical_energy = self._compute_text_energy(
            pixel_array, brightness_threshold
        )
        alignment_score = self._compute_alignment_score(horizontal_energy, vertical_energy)

        if alignment_score > 1.0:
            return OrientationResult(
                angle=0,
                method=f"primary(score={alignment_score:.2f},aligned)",
                reliable=True,
            )

        rotated_array = np.rot90(pixel_array, k=1)
        rotated_horizontal_energy, rotated_vertical_energy = self._compute_text_energy(
            rotated_array, brightness_threshold
        )
        rotated_alignment_score = self._compute_alignment_score(
            rotated_horizontal_energy, rotated_vertical_energy
        )

        if rotated_alignment_score > alignment_score:
            return OrientationResult(
                angle=90,
                method=(
                    f"primary(score={alignment_score:.2f}"
                    f"->90cw:{rotated_alignment_score:.2f})"
                ),
                reliable=True,
            )

        return OrientationResult(
            angle=270,
            method=f"primary(score={alignment_score:.2f}->270cw)",
            reliable=True,
        )

    @staticmethod
    def _compute_text_energy(
        pixel_array: np.ndarray, threshold: float
    ) -> tuple[float, float]:
        binary_mask = (pixel_array < threshold).astype(np.float32)
        horizontal_projection = binary_mask.sum(axis=1)
        vertical_projection = binary_mask.sum(axis=0)
        horizontal_energy = float(np.mean(np.diff(horizontal_projection) ** 2))
        vertical_energy = float(np.mean(np.diff(vertical_projection) ** 2))
        return horizontal_energy, vertical_energy

    @staticmethod
    def _compute_alignment_score(horizontal_energy: float, vertical_energy: float) -> float:
        return horizontal_energy / (vertical_energy + ENERGY_EPSILON)
