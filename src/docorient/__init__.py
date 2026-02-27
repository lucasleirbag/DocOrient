from docorient._version import __version__
from docorient.batch.processor import process_directory
from docorient.config import OrientationConfig
from docorient.correction import correct_document_pages, correct_image
from docorient.detection.base import DetectionEngine
from docorient.detection.engine import DetectionPipeline, detect_orientation
from docorient.detection.primary import PrimaryEngine
from docorient.detection.secondary import SecondaryEngine
from docorient.exceptions import (
    BatchProcessingError,
    CorrectionError,
    DetectionError,
    DocorientError,
    TesseractNotAvailableError,
)
from docorient.types import (
    BatchSummary,
    CorrectionResult,
    OrientationResult,
    PageResult,
)

__all__ = [
    "BatchProcessingError",
    "BatchSummary",
    "CorrectionError",
    "CorrectionResult",
    "DetectionEngine",
    "DetectionError",
    "DetectionPipeline",
    "DocorientError",
    "OrientationConfig",
    "OrientationResult",
    "PageResult",
    "PrimaryEngine",
    "SecondaryEngine",
    "TesseractNotAvailableError",
    "__version__",
    "correct_document_pages",
    "correct_image",
    "detect_orientation",
    "process_directory",
]
