from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from docorient.config import OrientationConfig
from docorient.types import OrientationResult

MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_onnx_session_cache: object | None = None
_onnx_available: bool | None = None


def _check_onnxruntime_available() -> bool:
    global _onnx_available
    if _onnx_available is None:
        try:
            import onnxruntime  # noqa: F401
            _onnx_available = True
        except ImportError:
            _onnx_available = False
    return _onnx_available


def _get_model_path() -> Path:
    model_ref = resources.files("docorient.models").joinpath("orientation_detector.onnx")
    return Path(str(model_ref))


def _get_onnx_session() -> object:
    global _onnx_session_cache
    if _onnx_session_cache is None:
        import onnxruntime as ort
        model_path = _get_model_path()
        _onnx_session_cache = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    return _onnx_session_cache


def _preprocess_image(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize((256, 256), Image.LANCZOS)

    crop_left = (256 - MODEL_INPUT_SIZE) // 2
    crop_top = (256 - MODEL_INPUT_SIZE) // 2
    cropped_image = resized_image.crop((
        crop_left,
        crop_top,
        crop_left + MODEL_INPUT_SIZE,
        crop_top + MODEL_INPUT_SIZE,
    ))

    pixel_array = np.array(cropped_image, dtype=np.float32) / 255.0
    normalized_array = (pixel_array - IMAGENET_MEAN) / IMAGENET_STD
    transposed_array = normalized_array.transpose(2, 0, 1)
    return np.expand_dims(transposed_array, axis=0)


def _classify_upside_down(image: Image.Image) -> tuple[bool, float]:
    session = _get_onnx_session()
    input_name = session.get_inputs()[0].name
    input_tensor = _preprocess_image(image)
    outputs = session.run(None, {input_name: input_tensor})
    logit = float(outputs[0][0][0])
    probability = 1.0 / (1.0 + np.exp(-logit))
    return logit > 0.0, probability


class FlipClassifierEngine:
    @property
    def name(self) -> str:
        return "flip_classifier"

    @staticmethod
    def is_available() -> bool:
        if not _check_onnxruntime_available():
            return False
        return _get_model_path().exists()

    def detect(
        self, image: Image.Image, config: OrientationConfig
    ) -> OrientationResult | None:
        if not self.is_available():
            return None

        is_flipped, confidence = _classify_upside_down(image)

        effective_confidence = confidence if is_flipped else (1.0 - confidence)
        if effective_confidence < config.flip_confidence_threshold:
            return None

        if is_flipped:
            return OrientationResult(
                angle=180,
                method=f"flip_classifier(flipped,conf={confidence:.2f})",
                reliable=True,
            )

        return OrientationResult(
            angle=0,
            method=f"flip_classifier(upright,conf={confidence:.2f})",
            reliable=True,
        )


def is_flip_classifier_available() -> bool:
    return FlipClassifierEngine.is_available()
