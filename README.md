# docorient

[![PyPI version](https://img.shields.io/pypi/v/docorient.svg)](https://pypi.org/project/docorient/)
[![Python](https://img.shields.io/pypi/pyversions/docorient.svg)](https://pypi.org/project/docorient/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Document image orientation detection and correction.

Automatically detects and fixes rotation (0°, 90°, 180°, 270°) in scanned document images using a two-stage detection pipeline, with support for multi-page majority voting and parallel batch processing.

---

## Installation

```bash
pip install docorient
```

To enable full 180° detection:

```bash
pip install docorient[ocr]
```

> **Note:** The `[ocr]` extra requires [Tesseract](https://github.com/tesseract-ocr/tesseract) installed on your system.
>
> - **macOS:** `brew install tesseract`
> - **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
> - **Windows:** [installer](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Quick Start

### Detect orientation

```python
from PIL import Image
from docorient import detect_orientation

image = Image.open("document.jpg")
result = detect_orientation(image)

print(result.angle)    # 0, 90, 180, or 270
print(result.reliable) # True
```

### Correct a single image

```python
from docorient import correct_image

corrected = correct_image(image)
corrected.save("fixed.jpg")
```

### Correct with metadata

```python
from docorient import correct_image

result = correct_image(image, return_metadata=True)
print(result.orientation.angle)
result.image.save("fixed.jpg")
```

### Correct a multi-page document (majority voting)

```python
from docorient import correct_document_pages
from PIL import Image

pages = [Image.open(f"page_{i}.jpg") for i in range(5)]
results = correct_document_pages(pages)

for i, result in enumerate(results):
    result.image.save(f"fixed_{i}.jpg")
```

### Batch process a directory

> **macOS / Windows:** `process_directory` uses multiprocessing internally.
> Always call it inside `if __name__ == "__main__":` when running as a script.

```python
from docorient import process_directory, OrientationConfig

if __name__ == "__main__":
    config = OrientationConfig(workers=4, output_quality=95)
    summary = process_directory("./scans", output_dir="./fixed", config=config)

    print(f"Corrected: {summary.corrected}/{summary.total_pages}")
    print(f"Errors:    {summary.errors}")
```

### CLI

```bash
docorient ./scans --output ./fixed
docorient ./scans --output ./fixed --workers 4 --quality 95
docorient ./scans --no-secondary --limit 100
docorient ./scans --dry-run
docorient --version
```

---

## Configuration Reference

```python
from docorient import OrientationConfig

config = OrientationConfig(
    secondary_confidence_threshold=2.0,  # Minimum confidence for secondary engine (default: 2.0)
    output_quality=92,                   # JPEG output quality 1-100 (default: 92)
    secondary_max_dimension=1200,        # Max image size for secondary engine (default: 1200)
    primary_max_dimension=800,           # Max image size for primary engine (default: 800)
    workers=4,                           # Parallel workers; None = cpu_count-2 (default: None)
    resume_enabled=True,                 # Resume interrupted batch jobs (default: True)
    supported_extensions=(               # File extensions to process
        ".jpg", ".jpeg", ".png",
        ".tiff", ".tif", ".bmp",
        ".gif", ".webp",
    ),
)
```

---

## API Reference

### `detect_orientation(image, config=None) → OrientationResult`

Detects the orientation of a document image without modifying it.

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image.Image` | Image to analyze |
| `config` | `OrientationConfig \| None` | Configuration (uses defaults if None) |

**Returns** `OrientationResult`:

| Field | Type | Description |
|---|---|---|
| `angle` | `int` | Detected angle: `0`, `90`, `180`, or `270` |
| `method` | `str` | Internal detection trace |
| `reliable` | `bool` | Whether the detection is considered reliable |

---

### `correct_image(image, *, config=None, return_metadata=False)`

Detects and corrects the orientation of a single image.

- `return_metadata=False` → returns `PIL.Image.Image`
- `return_metadata=True` → returns `CorrectionResult`

**`CorrectionResult`** fields: `image: PIL.Image.Image`, `orientation: OrientationResult`

---

### `correct_document_pages(pages, *, config=None) → list[CorrectionResult]`

Corrects orientation for all pages of a multi-page document with majority voting.

---

### `process_directory(input_dir, *, output_dir=None, config=None, limit=0, show_progress=True) → BatchSummary`

Processes all images in a directory in parallel.

**`BatchSummary`** fields:

| Field | Type | Description |
|---|---|---|
| `input_directory` | `str` | Resolved input path |
| `output_directory` | `str` | Resolved output path |
| `total_files` | `int` | Number of source documents |
| `total_pages` | `int` | Total images processed |
| `already_correct` | `int` | Pages at 0° (no correction needed) |
| `corrected` | `int` | Pages that were rotated |
| `corrected_by_majority` | `int` | Pages corrected via majority voting |
| `errors` | `int` | Pages that failed |
| `pages` | `tuple[PageResult, ...]` | Per-page results |

---

## Multi-page Document Grouping

`process_directory` automatically groups images by source document using the filename pattern `<name>_p<number>.<ext>`:

```
contrato_p1.jpg  ┐
contrato_p2.jpg  ├─► grouped as "contrato" → majority voting applied
contrato_p3.jpg  ┘

edital_p1.jpg    ┐
edital_p2.jpg    ┘─► grouped as "edital"
```

Files that don't match the pattern are treated as single-page documents.

---

## Supported Formats

Any format readable by Pillow: JPEG, PNG, TIFF, BMP, GIF, WebP, and more.

---

## Resume Support

Long batch jobs can be safely interrupted with `Ctrl+C`. Progress is saved to a `_orientation_done.log` file in the output directory. Re-running resumes from where it stopped.

To disable: `OrientationConfig(resume_enabled=False)`

---

## Exceptions

```python
from docorient import (
    DocorientError,
    DetectionError,
    CorrectionError,
    BatchProcessingError,
    TesseractNotAvailableError,
)
```

---

## Development

```bash
git clone https://github.com/lucasleirbag/docorient.git
cd docorient
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ocr]"
pytest tests/ -v
ruff check src/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE)
