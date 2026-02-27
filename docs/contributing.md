# Contributing

## Setup

```bash
git clone https://github.com/lucasleirbag/DocOrient.git
cd DocOrient
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=docorient --cov-report=term-missing
```

## Linting

```bash
ruff check src/ tests/
ruff check src/ tests/ --fix
```

## Code Standards

- No comments — code must be self-explanatory through naming
- Descriptive variable names — never single-letter variables in non-trivial scopes
- Full type hints on all function signatures
- Functions must do exactly one thing, max ~30 lines of logic
- Configuration is always passed as a parameter

## Retraining the FlipClassifier Model

The CNN model can be retrained with updated or expanded datasets:

```bash
pip install torch torchvision datasets onnxscript
python scripts/training/train_orientation_cnn.py
```

The script automatically downloads and caches images from 5 public datasets to `data/image_cache/`.
On subsequent runs, the cache is reused (no re-download). Training uses MPS (Apple Silicon)
or CUDA when available. The exported ONNX model is saved to `models/orientation_detector.onnx`.

After retraining, copy the model to the package and validate:

```bash
cp models/orientation_detector.onnx src/docorient/models/orientation_detector.onnx
python scripts/training/export_and_validate.py
pytest tests/ -v
```

## Publishing a New Version

1. Update `version` in `pyproject.toml` and `src/docorient/_version.py`
2. Add an entry to `CHANGELOG.md`
3. Rebuild and publish:

```bash
rm -rf dist/
python -m build
twine check dist/*
twine upload dist/*
```
