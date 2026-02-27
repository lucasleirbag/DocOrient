from __future__ import annotations

from dataclasses import dataclass

from PIL import Image as PILImage


@dataclass(frozen=True, slots=True)
class OrientationResult:
    angle: int
    method: str
    reliable: bool


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    image: PILImage.Image
    orientation: OrientationResult


@dataclass(frozen=True, slots=True)
class PageResult:
    source_file: str
    page_number: int
    image_name: str
    input_path: str
    output_path: str
    orientation: OrientationResult
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchSummary:
    input_directory: str
    output_directory: str
    total_files: int
    total_pages: int
    already_correct: int
    corrected: int
    corrected_by_majority: int
    errors: int
    pages: tuple[PageResult, ...]
