from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from threading import Lock

from .labels import COFFEE_DISEASE_LABELS

DEFAULT_INPUT_SIZE = 224


def _find_project_root(start: Path) -> Path:
    markers = ("pyproject.toml", "requirements.txt", ".git")
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return start


def _default_model_path() -> Path:
    env_path = os.environ.get("COFFEE_MODEL_PATH")
    if env_path:
        return Path(env_path)
    project_root = _find_project_root(Path(__file__).resolve().parent)
    return project_root / "models" / "keras_model.h5"


DEFAULT_MODEL_PATH = _default_model_path()


class CoffeeDiseaseClassifier:
    name = "EfficientNetB0 (Coffee Disease)"
    _model = None
    _lock = Lock()

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels: list[str] | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        self.model_path = (
            DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
        )
        self.labels = labels or COFFEE_DISEASE_LABELS
        self.input_size = (input_size, input_size)

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from tensorflow.keras.models import load_model
                    import tensorflow as tf

                    print("\n🔄 Loading coffee disease model...")
                    print("📁 Path:", self.model_path)
                    print("📌 Exists:", self.model_path.exists())

                    if not self.model_path.exists():
                        raise FileNotFoundError(f"Model not found at {self.model_path}")

                    # 🔥 Fix for DepthwiseConv2D / old TF models
                    self._model = load_model(
                        self.model_path,
                        compile=False,
                        custom_objects={"DepthwiseConv2D": tf.keras.layers.DepthwiseConv2D},
                    )

                    print("✅ Coffee disease model loaded successfully!\n")

        return self._model

    def predict(self, raw_bytes: bytes, top: int = 5) -> list[dict[str, float | str]]:
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        from tensorflow.keras.applications.efficientnet import preprocess_input
        from tensorflow.keras.preprocessing import image

        pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB").resize(self.input_size)
        image_array = image.img_to_array(pil_image)
        batch = np.expand_dims(image_array, axis=0)
        batch = preprocess_input(batch)

        predictions = self._get_model().predict(batch, verbose=0)
        scores = np.squeeze(predictions)

        if scores.ndim != 1:
            raise ValueError("Unexpected model output shape.")

        if scores.shape[0] != len(self.labels):
            raise ValueError("Model output does not match labels.")

        total = float(scores.sum())
        if scores.min() < 0 or not 0.9 <= total <= 1.1:
            scores = tf.nn.softmax(scores).numpy()

        top = min(top, len(self.labels))
        top_indices = np.argsort(scores)[::-1][:top]

        return [
            {
                "id": int(index),
                "label": self.labels[index],
                "confidence": float(scores[index]),
            }
            for index in top_indices
        ]


class VGG16Classifier:
    name = "VGG16 (ImageNet)"
    _model = None
    _lock = Lock()

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    from tensorflow.keras.applications.vgg16 import VGG16

                    print("⚠️ Loading fallback VGG16 model...")
                    cls._model = VGG16(weights="imagenet")

        return cls._model

    def predict(self, raw_bytes: bytes, top: int = 5) -> list[dict[str, float | str]]:
        import numpy as np
        from PIL import Image
        from tensorflow.keras.applications.vgg16 import (
            decode_predictions,
            preprocess_input,
        )
        from tensorflow.keras.preprocessing import image

        pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB").resize((224, 224))
        image_array = image.img_to_array(pil_image)
        batch = np.expand_dims(image_array, axis=0)
        batch = preprocess_input(batch)

        predictions = self._get_model().predict(batch, verbose=0)
        decoded = decode_predictions(predictions, top=top)[0]

        return [
            {
                "id": imagenet_id,
                "label": label.replace("_", " "),
                "confidence": float(score),
            }
            for imagenet_id, label, score in decoded
        ]


class ModelRouter:
    def __init__(
        self,
        primary: CoffeeDiseaseClassifier,
        fallback: VGG16Classifier,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active = primary
        self.notice: str | None = None

    @property
    def name(self) -> str:
        return self._active.name

    @property
    def default_name(self) -> str:
        return self._primary.name

    def predict(self, raw_bytes: bytes, top: int = 5) -> list[dict[str, float | str]]:
        try:
            predictions = self._primary.predict(raw_bytes, top=top)
            self._active = self._primary
            self.notice = None
            return predictions

        except Exception as e:
            print("❌ Primary model failed:", e)
            self._active = self._fallback
            self.notice = "Primary model failed. Using fallback ImageNet model."
            return self._fallback.predict(raw_bytes, top=top)


# 🔥 Initialize classifier
classifier = ModelRouter(
    primary=CoffeeDiseaseClassifier(),
    fallback=VGG16Classifier(),
)
