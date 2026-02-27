from __future__ import annotations

from PIL import Image

from docorient.config import OrientationConfig
from docorient.detection.base import DetectionEngine
from docorient.detection.primary import PrimaryEngine
from docorient.detection.secondary import SecondaryEngine
from docorient.types import OrientationResult


class DetectionPipeline:
    def __init__(self, engines: list[DetectionEngine] | None = None) -> None:
        self.engines: list[DetectionEngine] = engines or [PrimaryEngine(), SecondaryEngine()]

    def run(self, image: Image.Image, config: OrientationConfig) -> OrientationResult:
        primary_result: OrientationResult | None = None

        for engine in self.engines:
            result = engine.detect(image, config)

            if result is None:
                continue

            if engine.name == "primary":
                primary_result = result
                if result.angle in (90, 270):
                    return result
                continue

            if primary_result is not None:
                combined_method = f"{result.method},{primary_result.method}"
            else:
                combined_method = result.method
            return OrientationResult(
                angle=result.angle,
                method=combined_method,
                reliable=True,
            )

        if primary_result is not None:
            return primary_result

        return OrientationResult(angle=0, method="none", reliable=False)


_default_pipeline = DetectionPipeline()


def detect_orientation(
    image: Image.Image,
    config: OrientationConfig | None = None,
) -> OrientationResult:
    effective_config = config or OrientationConfig()
    return _default_pipeline.run(image, effective_config)
