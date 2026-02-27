# Changelog

## 0.3.1 (2026-02-27)

- Fix: author metadata corrected to Lucas Gabriel Vaz
- Remove `tesseract` from PyPI keywords

## 0.3.0 (2026-02-27)

- **Breaking:** detection engines refactored to class-based architecture with `DetectionEngine` Protocol
- Introduce `DetectionPipeline` for extensible engine orchestration
- Introduce `PrimaryEngine` and `SecondaryEngine` classes
- Extract `rotation.py` and `voting.py` as standalone modules
- Encapsulate worker state in `WorkerContext` dataclass
- Transform `_imaging.py` functions into `ImageIO` class
- Decompose `process_directory` into focused sub-functions
- Apply custom exceptions (`DetectionError`, `CorrectionError`, `BatchProcessingError`) throughout codebase
- Export `DetectionEngine`, `DetectionPipeline`, `PrimaryEngine`, `SecondaryEngine` in public API

## 0.2.0 (2026-02-26)

- **Breaking:** renamed config params `osd_confidence_threshold` → `secondary_confidence_threshold`, `max_osd_dimension` → `secondary_max_dimension`, `projection_target_dimension` → `primary_max_dimension`
- **Breaking:** renamed CLI flag `--no-ocr` → `--no-secondary`
- Internal engines renamed to `primary` and `secondary`
- Updated `OrientationResult.method` trace strings

## 0.1.1 (2026-02-26)

- Docs: added `if __name__ == "__main__":` note for `process_directory` on macOS/Windows

## 0.1.0 (2026-02-25)

- Initial release
- Primary engine for 90°/270° detection
- Optional secondary engine for 180° detection
- Single image and batch directory processing
- Multi-page majority voting
- Resumable batch processing
- CLI interface
