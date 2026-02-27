from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Any

from docorient._imaging import ImageIO
from docorient.batch.scanner import ScannedPage
from docorient.config import OrientationConfig
from docorient.detection.engine import detect_orientation
from docorient.exceptions import CorrectionError, DetectionError
from docorient.rotation import apply_rotation
from docorient.types import OrientationResult, PageResult
from docorient.voting import apply_majority_voting


@dataclass
class WorkerContext:
    progress_counter: Any
    progress_lock: Any
    resume_log_path: str
    config: OrientationConfig


_worker_context: WorkerContext | None = None


def initialize_worker(
    counter: multiprocessing.Value,
    lock: multiprocessing.Lock,
    resume_log_path: str,
    config_dict: dict[str, Any],
) -> None:
    global _worker_context
    _worker_context = WorkerContext(
        progress_counter=counter,
        progress_lock=lock,
        resume_log_path=resume_log_path,
        config=OrientationConfig(**config_dict),
    )


def _process_single_source(
    source_file_name: str,
    pages: list[ScannedPage],
    config: OrientationConfig,
) -> list[PageResult]:
    valid_pages = list(pages)
    detection_results: list[OrientationResult] = []
    page_errors: dict[int, str] = {}

    for page_index, scanned_page in enumerate(valid_pages):
        try:
            image = ImageIO.open_as_rgb(scanned_page.image_path)
            orientation = detect_orientation(image, config=config)
            detection_results.append(orientation)
            image.close()
        except Exception as original_error:
            wrapped = DetectionError(
                f"Detection failed for {scanned_page.image_name}: {original_error}"
            )
            detection_results.append(OrientationResult(angle=0, method="error", reliable=False))
            page_errors[page_index] = str(wrapped)

    if len(valid_pages) > 1:
        detection_results = apply_majority_voting(detection_results)

    page_results: list[PageResult] = []

    for page_index, (scanned_page, orientation) in enumerate(zip(valid_pages, detection_results)):
        error_message = page_errors.get(page_index)

        if error_message is None:
            try:
                image = ImageIO.open_as_rgb(scanned_page.image_path)
                corrected_image = apply_rotation(image, orientation.angle)
                output_format = ImageIO.resolve_format(scanned_page.output_path)
                ImageIO.save(
                    corrected_image,
                    scanned_page.output_path,
                    output_format=output_format,
                    quality=config.output_quality,
                )
                corrected_image.close()
                image.close()
            except Exception as original_error:
                wrapped = CorrectionError(
                    f"Correction failed for {scanned_page.image_name}: {original_error}"
                )
                error_message = str(wrapped)

        page_results.append(
            PageResult(
                source_file=scanned_page.source_file,
                page_number=scanned_page.page_number,
                image_name=scanned_page.image_name,
                input_path=scanned_page.image_path,
                output_path=scanned_page.output_path,
                orientation=orientation,
                error=error_message,
            )
        )

    return page_results


def _record_completion(source_file_name: str) -> None:
    assert _worker_context is not None
    with _worker_context.progress_lock:
        _worker_context.progress_counter.value += 1
        try:
            with open(_worker_context.resume_log_path, "a") as resume_log:
                resume_log.write(source_file_name + "\n")
                resume_log.flush()
        except OSError:
            pass


def process_batch(
    batch: list[tuple[str, list[ScannedPage]]],
) -> list[tuple[str, list[PageResult]]]:
    assert _worker_context is not None
    batch_results: list[tuple[str, list[PageResult]]] = []

    for source_file_name, scanned_pages in batch:
        page_results = _process_single_source(
            source_file_name, scanned_pages, _worker_context.config
        )
        batch_results.append((source_file_name, page_results))
        _record_completion(source_file_name)

    return batch_results
