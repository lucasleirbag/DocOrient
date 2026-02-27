from __future__ import annotations

from typing import Protocol

from PIL import Image

from docorient.config import OrientationConfig
from docorient.types import OrientationResult


class DetectionEngine(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, image: Image.Image, config: OrientationConfig) -> OrientationResult | None: ...
