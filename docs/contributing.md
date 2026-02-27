# Contributing

## Setup

```bash
git clone https://github.com/lucasleirbag/docorient.git
cd docorient
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ocr]"
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
