from __future__ import annotations

from typing import overload

from PIL import Image

from docorient.config import OrientationConfig
from docorient.detection.engine import detect_orientation
from docorient.exceptions import CorrectionError
from docorient.rotation import apply_rotation
from docorient.types import CorrectionResult
from docorient.voting import apply_majority_voting


@overload
def correct_image(
    image: Image.Image,
    *,
    config: OrientationConfig | None = ...,
    return_metadata: bool = False,
) -> Image.Image: ...


@overload
def correct_image(
    image: Image.Image,
    *,
    config: OrientationConfig | None = ...,
    return_metadata: bool = True,
) -> CorrectionResult: ...


def correct_image(
    image: Image.Image,
    *,
    config: OrientationConfig | None = None,
    return_metadata: bool = False,
) -> Image.Image | CorrectionResult:
    try:
        orientation = detect_orientation(image, config=config)
        corrected_image = apply_rotation(image, orientation.angle)
    except Exception as original_error:
        raise CorrectionError(f"Failed to correct image: {original_error}") from original_error

    if return_metadata:
        return CorrectionResult(image=corrected_image, orientation=orientation)
    return corrected_image


def correct_document_pages(
    pages: list[Image.Image],
    *,
    config: OrientationConfig | None = None,
) -> list[CorrectionResult]:
    effective_config = config or OrientationConfig()

    detection_results = [
        detect_orientation(page_image, config=effective_config) for page_image in pages
    ]

    if len(pages) > 1:
        detection_results = apply_majority_voting(detection_results)

    correction_results = []
    for page_image, orientation in zip(pages, detection_results):
        corrected_page = apply_rotation(page_image, orientation.angle)
        correction_results.append(
            CorrectionResult(image=corrected_page, orientation=orientation)
        )

    return correction_results
