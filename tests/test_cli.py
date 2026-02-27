import subprocess
import sys
from pathlib import Path


class TestCLI:
    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "docorient.cli", "--version"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0
        assert "docorient" in result.stdout

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "docorient.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0
        assert "input_dir" in result.stdout

    def test_help_has_no_internal_references(self):
        result = subprocess.run(
            [sys.executable, "-m", "docorient.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert "tesseract" not in result.stdout.lower()
        assert "projection" not in result.stdout.lower()

    def test_invalid_directory(self, tmp_path):
        nonexistent = str(tmp_path / "nonexistent")
        result = subprocess.run(
            [sys.executable, "-m", "docorient.cli", nonexistent],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode != 0

    def test_dry_run(self, populated_images_dir):
        result = subprocess.run(
            [sys.executable, "-m", "docorient.cli", str(populated_images_dir), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
