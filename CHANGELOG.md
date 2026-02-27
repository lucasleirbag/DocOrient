# Changelog

## 0.4.1 (2026-02-27)

- Clean up all legacy references from documentation and CI
- Update README description with pipeline accuracy metrics
- Add retraining instructions to contributing guide

## 0.4.0 (2026-02-27)

- **Breaking:** remove `SecondaryEngine`, `[ocr]` extra
- **Breaking:** remove `secondary_confidence_threshold` and `secondary_max_dimension` from `OrientationConfig`
- Add `FlipClassifierEngine`: MobileNetV2-based ONNX neural network for 180° detection
- ONNX model (~8.5 MB) ships embedded in the package — no external binaries needed
- Model trained on 3,000+ diverse document images from 5 public datasets (FUNSD, Invoices & Receipts, CORD-v2, DocLayNet, DocumentVQA)
- CNN standalone accuracy: 96.8% — combined pipeline accuracy: 98% across all 4 orientations
- Inference latency: ~5ms per image (CNN) / ~20ms per image (full pipeline)
- Add `flip_confidence_threshold` to `OrientationConfig`
- `onnxruntime` is now a core dependency (no longer optional)
- Detection pipeline now resolves all 4 angles (0°, 90°, 180°, 270°) without external tools
- Pipeline: PrimaryEngine detects axis (H/V), FlipClassifier resolves flip within each axis
- Update CLI: remove `--no-secondary` and `--confidence` flags
- Bump status to "Development Status :: 4 - Beta"

## 0.3.2 (2026-02-27)

- Fix: update project URLs to correct GitHub repository
- Add changelog link to PyPI metadata

## 0.3.1 (2026-02-27)

- Fix: author metadata corrected to Lucas Gabriel Vaz
- Update PyPI keywords

## 0.3.0 (2026-02-27)

- **Breaking:** detection engines refactored to class-based architecture with `DetectionEngine` Protocol
- Introduce `DetectionPipeline` for extensible engine orchestration
- Introduce `PrimaryEngine` and `SecondaryEngine` classes
- Extract `rotation.py` and `voting.py` as standalone modules
- Encapsulate worker state in `WorkerContext` dataclass
- Transform `_imaging.py` functions into `ImageIO` class
- Decompose `process_directory` into focused sub-functions
- Apply custom exceptions throughout codebase
- Export `DetectionEngine`, `DetectionPipeline`, `PrimaryEngine` in public API

## 0.2.0 (2026-02-26)

- **Breaking:** renamed config params and CLI flags for internal engines
- Updated `OrientationResult.method` trace strings

## 0.1.1 (2026-02-26)

- Docs: added `if __name__ == "__main__":` note for `process_directory` on macOS/Windows

## 0.1.0 (2026-02-25)

- Initial release
- Primary engine for 90°/270° detection
- Single image and batch directory processing
- Multi-page majority voting
- Resumable batch processing
- CLI interface
