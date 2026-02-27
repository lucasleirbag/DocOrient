from __future__ import annotations

from pathlib import Path

from PIL import Image

FORMAT_MAPPING: dict[str, str] = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".tiff": "TIFF",
    ".tif": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".webp": "WEBP",
}


class ImageIO:
    @staticmethod
    def open_as_rgb(image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def downscale(image: Image.Image, max_dimension: int) -> Image.Image:
        image_width, image_height = image.size
        largest_side = max(image_width, image_height)

        if largest_side <= max_dimension:
            return image

        scale_factor = max_dimension / largest_side
        target_width = int(image_width * scale_factor)
        target_height = int(image_height * scale_factor)
        return image.resize((target_width, target_height), Image.LANCZOS)

    @staticmethod
    def save(
        image: Image.Image,
        output_path: str | Path,
        output_format: str = "JPEG",
        quality: int = 92,
    ) -> None:
        image.save(output_path, output_format, quality=quality)

    @staticmethod
    def resolve_format(file_path: str | Path) -> str:
        extension = Path(file_path).suffix.lower()
        return FORMAT_MAPPING.get(extension, "JPEG")
