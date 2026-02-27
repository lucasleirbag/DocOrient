from __future__ import annotations

import warnings

from PIL import Image

from docorient._imaging import ImageIO
from docorient.config import OrientationConfig
from docorient.types import OrientationResult

_secondary_engine_available: bool | None = None


class SecondaryEngine:
    @property
    def name(self) -> str:
        return "secondary"

    @staticmethod
    def is_available() -> bool:
        global _secondary_engine_available
        if _secondary_engine_available is None:
            try:
                import pytesseract  # noqa: F401

                _secondary_engine_available = True
            except ImportError:
                _secondary_engine_available = False
        return _secondary_engine_available

    def detect(
        self, image: Image.Image, config: OrientationConfig
    ) -> OrientationResult | None:
        if not self.is_available():
            warnings.warn(
                "Secondary engine not available. 180° detection is disabled. "
                "Install with: pip install docorient[ocr]",
                UserWarning,
                stacklevel=2,
            )
            return None

        downscaled_image = ImageIO.downscale(image, config.secondary_max_dimension)
        detected_angle, detection_confidence = self._run_engine(downscaled_image)

        if downscaled_image is not image:
            downscaled_image.close()

        if detection_confidence < config.secondary_confidence_threshold:
            return None

        if detected_angle not in (90, 180, 270):
            return None

        return OrientationResult(
            angle=detected_angle,
            method=f"secondary(angle={detected_angle},conf={detection_confidence:.1f})",
            reliable=True,
        )

    @staticmethod
    def _run_engine(image: Image.Image) -> tuple[int, float]:
        import pytesseract

        try:
            result = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            detected_angle = int(result.get("orientation", 0))
            detection_confidence = float(result.get("orientation_conf", 0.0))
            return detected_angle, detection_confidence
        except pytesseract.TesseractError:
            return 0, 0.0


def is_secondary_engine_available() -> bool:
    return SecondaryEngine.is_available()
