from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field

DEFAULT_SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
)

RESUME_LOG_FILENAME = "_orientation_done.log"


def _default_worker_count() -> int:
    return max(1, multiprocessing.cpu_count() - 2)


@dataclass(frozen=True, slots=True)
class OrientationConfig:
    flip_confidence_threshold: float = 0.50
    output_quality: int = 92
    primary_max_dimension: int = 800
    workers: int | None = None
    resume_enabled: bool = True
    supported_extensions: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_SUPPORTED_EXTENSIONS
    )

    @property
    def effective_workers(self) -> int:
        if self.workers is not None:
            return max(1, self.workers)
        return _default_worker_count()
