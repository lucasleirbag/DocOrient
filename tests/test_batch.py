from pathlib import Path

from docorient.batch.scanner import ScannedPage, scan_directory
from docorient.types import BatchSummary


class TestScanner:
    def test_scan_empty_directory(self, tmp_images_dir):
        output_dir = tmp_images_dir.parent / "output"
        output_dir.mkdir()
        result = scan_directory(
            tmp_images_dir, output_dir, supported_extensions=(".jpg",)
        )
        assert result == {}

    def test_scan_finds_jpg_images(self, populated_images_dir):
        output_dir = populated_images_dir.parent / "output"
        output_dir.mkdir()
        result = scan_directory(
            populated_images_dir, output_dir, supported_extensions=(".jpg",)
        )
        assert "doc001" in result
        assert len(result["doc001"]) == 3

    def test_scan_groups_by_source_file(self, tmp_images_dir, horizontal_text_image):
        for doc_name in ("docA", "docB"):
            for page_number in range(1, 3):
                image_name = f"{doc_name}_p{page_number}.jpg"
                horizontal_text_image.save(tmp_images_dir / image_name, "JPEG")

        output_dir = tmp_images_dir.parent / "output"
        output_dir.mkdir()
        result = scan_directory(
            tmp_images_dir, output_dir, supported_extensions=(".jpg",)
        )
        assert "docA" in result
        assert "docB" in result
        assert len(result["docA"]) == 2
        assert len(result["docB"]) == 2

    def test_scan_respects_limit(self, populated_images_dir):
        output_dir = populated_images_dir.parent / "output"
        output_dir.mkdir()
        result = scan_directory(
            populated_images_dir, output_dir, supported_extensions=(".jpg",), limit=1
        )
        total_pages = sum(len(pages) for pages in result.values())
        assert total_pages == 1

    def test_scan_respects_extensions(self, tmp_images_dir, horizontal_text_image):
        horizontal_text_image.save(tmp_images_dir / "test.jpg", "JPEG")
        horizontal_text_image.save(tmp_images_dir / "test.png", "PNG")

        output_dir = tmp_images_dir.parent / "output"
        output_dir.mkdir()

        jpg_only = scan_directory(
            tmp_images_dir, output_dir, supported_extensions=(".jpg",)
        )
        assert sum(len(pages) for pages in jpg_only.values()) == 1

        both = scan_directory(
            tmp_images_dir, output_dir, supported_extensions=(".jpg", ".png")
        )
        assert sum(len(pages) for pages in both.values()) == 2

    def test_scanned_page_has_correct_paths(self, populated_images_dir):
        output_dir = populated_images_dir.parent / "output"
        output_dir.mkdir()
        result = scan_directory(
            populated_images_dir, output_dir, supported_extensions=(".jpg",)
        )
        first_page = result["doc001"][0]
        assert isinstance(first_page, ScannedPage)
        assert first_page.source_file == "doc001"
        assert str(populated_images_dir) in first_page.image_path
        assert str(output_dir) in first_page.output_path


class TestProcessDirectory:
    def test_process_empty_directory(self, tmp_images_dir):
        from docorient.batch.processor import process_directory

        summary = process_directory(
            tmp_images_dir,
            output_dir=tmp_images_dir.parent / "output",
            show_progress=False,
        )
        assert isinstance(summary, BatchSummary)
        assert summary.total_pages == 0

    def test_process_with_images(self, populated_images_dir):
        from docorient.batch.processor import process_directory

        output_dir = populated_images_dir.parent / "output"
        summary = process_directory(
            populated_images_dir,
            output_dir=output_dir,
            show_progress=False,
        )
        assert isinstance(summary, BatchSummary)
        assert summary.total_pages == 3
        assert summary.errors == 0

    def test_output_files_are_created(self, populated_images_dir):
        from docorient.batch.processor import process_directory

        output_dir = populated_images_dir.parent / "output"
        process_directory(
            populated_images_dir,
            output_dir=output_dir,
            show_progress=False,
        )
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3

    def test_auto_generated_output_dir(self, populated_images_dir):
        from docorient.batch.processor import process_directory

        summary = process_directory(
            populated_images_dir,
            show_progress=False,
        )
        assert Path(summary.output_directory).exists()
        assert summary.total_pages == 3
