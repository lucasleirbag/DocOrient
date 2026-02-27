import pytest
from PIL import Image, ImageDraw


def _create_text_lines_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    line_spacing = 30
    for y_position in range(20, height - 20, line_spacing):
        draw.line([(40, y_position), (width - 40, y_position)], fill=(0, 0, 0), width=2)
    return image


@pytest.fixture
def horizontal_text_image() -> Image.Image:
    return _create_text_lines_image(800, 600)


@pytest.fixture
def vertical_text_image() -> Image.Image:
    horizontal = _create_text_lines_image(800, 600)
    return horizontal.rotate(90, expand=True)


@pytest.fixture
def small_blank_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


@pytest.fixture
def tmp_images_dir(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    return input_dir


@pytest.fixture
def populated_images_dir(tmp_images_dir, horizontal_text_image):
    for page_index in range(3):
        image_name = f"doc001_p{page_index + 1}.jpg"
        horizontal_text_image.save(tmp_images_dir / image_name, "JPEG")
    return tmp_images_dir
