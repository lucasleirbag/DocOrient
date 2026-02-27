from __future__ import annotations

from PIL import Image

from docorient.config import OrientationConfig
from docorient.detection.base import DetectionEngine
from docorient.detection.flip_classifier import FlipClassifierEngine
from docorient.detection.primary import PrimaryEngine
from docorient.types import OrientationResult


def _build_default_engines() -> list[DetectionEngine]:
    return [PrimaryEngine(), FlipClassifierEngine()]


class DetectionPipeline:
    def __init__(self, engines: list[DetectionEngine] | None = None) -> None:
        self.engines: list[DetectionEngine] = engines or _build_default_engines()

    def run(self, image: Image.Image, config: OrientationConfig) -> OrientationResult:
        primary_result: OrientationResult | None = None

        for engine in self.engines:
            if engine.name == "primary":
                primary_result = engine.detect(image, config)
                continue

            if engine.name == "flip_classifier":
                flip_result = self._resolve_flip(engine, image, config, primary_result)
                if flip_result is not None:
                    return flip_result
                continue

        if primary_result is not None:
            return primary_result

        return OrientationResult(angle=0, method="none", reliable=False)

    @staticmethod
    def _resolve_flip(
        engine: DetectionEngine,
        image: Image.Image,
        config: OrientationConfig,
        primary_result: OrientationResult | None,
    ) -> OrientationResult | None:
        is_vertical = primary_result is not None and primary_result.angle in (90, 270)
        normalized_image = (
            image.transpose(Image.Transpose.ROTATE_90) if is_vertical else image
        )

        flip_result = engine.detect(normalized_image, config)

        if is_vertical and normalized_image is not image:
            normalized_image.close()

        if flip_result is None:
            return None

        if is_vertical:
            final_angle = 270 if flip_result.angle == 180 else 90
        else:
            final_angle = flip_result.angle

        combined_method = flip_result.method
        if primary_result is not None:
            combined_method = f"{flip_result.method},{primary_result.method}"

        return OrientationResult(
            angle=final_angle,
            method=combined_method,
            reliable=flip_result.reliable,
        )


_default_pipeline = DetectionPipeline()


def detect_orientation(
    image: Image.Image,
    config: OrientationConfig | None = None,
) -> OrientationResult:
    effective_config = config or OrientationConfig()
    return _default_pipeline.run(image, effective_config)
