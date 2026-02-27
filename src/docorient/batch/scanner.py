from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PAGE_PATTERN = re.compile(r"^(.+)_p(\d+)\.\w+$")


@dataclass(frozen=True, slots=True)
class ScannedPage:
    source_file: str
    page_number: int
    image_name: str
    image_path: str
    output_path: str


def scan_directory(
    input_directory: Path,
    output_directory: Path,
    supported_extensions: tuple[str, ...],
    limit: int = 0,
) -> dict[str, list[ScannedPage]]:
    all_image_paths = sorted(
        image_path
        for image_path in input_directory.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in supported_extensions
    )

    if limit > 0:
        all_image_paths = all_image_paths[:limit]

    pages_by_source: dict[str, list[ScannedPage]] = {}

    for image_path in all_image_paths:
        image_name = image_path.name
        page_match = PAGE_PATTERN.match(image_name)

        if page_match:
            source_file_name = page_match.group(1)
            page_number = int(page_match.group(2))
        else:
            source_file_name = image_path.stem
            page_number = 1

        output_path = output_directory / image_name

        scanned_page = ScannedPage(
            source_file=source_file_name,
            page_number=page_number,
            image_name=image_name,
            image_path=str(image_path),
            output_path=str(output_path),
        )

        pages_by_source.setdefault(source_file_name, []).append(scanned_page)

    return pages_by_source
