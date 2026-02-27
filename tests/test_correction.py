from PIL import Image

from docorient.correction import correct_document_pages, correct_image
from docorient.types import CorrectionResult, OrientationResult


class TestCorrectImage:
    def test_returns_pil_image_by_default(self, horizontal_text_image):
        result = correct_image(horizontal_text_image)
        assert isinstance(result, Image.Image)

    def test_returns_correction_result_with_metadata(self, horizontal_text_image):
        result = correct_image(horizontal_text_image, return_metadata=True)
        assert isinstance(result, CorrectionResult)
        assert isinstance(result.image, Image.Image)
        assert isinstance(result.orientation, OrientationResult)

    def test_horizontal_image_stays_same_size(self, horizontal_text_image):
        original_size = horizontal_text_image.size
        corrected = correct_image(horizontal_text_image)
        assert corrected.size == original_size

    def test_vertical_image_gets_rotated(self, vertical_text_image):
        original_width, original_height = vertical_text_image.size
        result = correct_image(vertical_text_image, return_metadata=True)
        if result.orientation.angle in (90, 270):
            corrected_width, corrected_height = result.image.size
            assert corrected_width == original_height
            assert corrected_height == original_width


class TestCorrectDocumentPages:
    def test_returns_list_of_correction_results(self, horizontal_text_image):
        pages = [horizontal_text_image.copy() for _ in range(3)]
        results = correct_document_pages(pages)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, CorrectionResult)

    def test_single_page_document(self, horizontal_text_image):
        results = correct_document_pages([horizontal_text_image])
        assert len(results) == 1
        assert isinstance(results[0], CorrectionResult)

    def test_empty_page_list(self):
        results = correct_document_pages([])
        assert results == []

    def test_majority_voting_with_consistent_pages(self, horizontal_text_image):
        pages = [horizontal_text_image.copy() for _ in range(5)]
        results = correct_document_pages(pages)
        angles = [result.orientation.angle for result in results]
        assert all(angle == angles[0] for angle in angles)
