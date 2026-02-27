from docorient.config import OrientationConfig
from docorient.detection.engine import detect_orientation
from docorient.detection.primary import PrimaryEngine
from docorient.types import OrientationResult


class TestPrimaryEngine:
    def setup_method(self):
        self.engine = PrimaryEngine()
        self.default_config = OrientationConfig()

    def test_aligned_image_returns_zero_angle(self, horizontal_text_image):
        result = self.engine.detect(horizontal_text_image, self.default_config)
        assert result.angle == 0
        assert result.reliable is True
        assert "primary" in result.method

    def test_rotated_image_returns_nonzero_angle(self, vertical_text_image):
        result = self.engine.detect(vertical_text_image, self.default_config)
        assert result.angle in (90, 270)
        assert result.reliable is True

    def test_blank_image_returns_result(self, small_blank_image):
        result = self.engine.detect(small_blank_image, self.default_config)
        assert isinstance(result, OrientationResult)
        assert result.angle in (0, 90, 270)

    def test_custom_target_dimension(self, horizontal_text_image):
        config = OrientationConfig(primary_max_dimension=400)
        result = self.engine.detect(horizontal_text_image, config)
        assert isinstance(result, OrientationResult)


class TestDetectionEngine:
    def test_aligned_image_detection(self, horizontal_text_image):
        result = detect_orientation(horizontal_text_image)
        assert result.angle == 0
        assert result.reliable is True

    def test_rotated_image_detection(self, vertical_text_image):
        result = detect_orientation(vertical_text_image)
        assert result.angle in (90, 270)

    def test_with_custom_config(self, horizontal_text_image):
        config = OrientationConfig(
            secondary_confidence_threshold=10.0,
            primary_max_dimension=400,
        )
        result = detect_orientation(horizontal_text_image, config=config)
        assert isinstance(result, OrientationResult)

    def test_result_has_all_fields(self, horizontal_text_image):
        result = detect_orientation(horizontal_text_image)
        assert hasattr(result, "angle")
        assert hasattr(result, "method")
        assert hasattr(result, "reliable")
        assert isinstance(result.angle, int)
        assert isinstance(result.method, str)
        assert isinstance(result.reliable, bool)
