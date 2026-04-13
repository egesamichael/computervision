from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from threading import Lock

from .labels import COFFEE_DISEASE_LABELS


def _default_input_size() -> int:
    raw = os.environ.get("COFFEE_MODEL_INPUT_SIZE")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 224


DEFAULT_INPUT_SIZE = _default_input_size()


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
    candidates = [
        project_root / "models" / "coffee_disease_efficientnetb0.keras",
        project_root / "models" / "keras_model.h5",
        project_root / "keras_model.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _looks_like_hdf5(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
        return signature == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


DEFAULT_MODEL_PATH = _default_model_path()


class CoffeeDiseaseClassifier:
    name = "EfficientNetB0 (Coffee Disease)"
    _model = None
    _lock = Lock()

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels: list[str] | None = None,
        input_size: int | None = None,
    ) -> None:
        self.model_path = (
            DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
        )
        self.labels = labels or COFFEE_DISEASE_LABELS
        resolved_input = DEFAULT_INPUT_SIZE if input_size is None else input_size
        self.input_size = (resolved_input, resolved_input)

    def _rebuild_sequential(self, model, tf):
        inputs = tf.keras.Input(
            shape=(self.input_size[0], self.input_size[1], 3)
        )
        x = inputs
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
        return tf.keras.Model(inputs, x, name=f"{model.name}_compat")

    def _ensure_model_compatible(self, model, tf):
        try:
            dummy = tf.zeros((1, self.input_size[0], self.input_size[1], 3))
            _ = model(dummy, training=False)
            return model
        except Exception as exc:
            message = str(exc)
            if (
                "expects 1 input(s), but it received 2 input tensors" in message
                and isinstance(model, tf.keras.Sequential)
            ):
                print("⚠️ Detected legacy model output mismatch. Rebuilding model for compatibility...")
                repaired = self._rebuild_sequential(model, tf)
                _ = repaired(dummy, training=False)
                return repaired
            raise

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from tensorflow.keras.models import load_model
                    import tensorflow as tf
                    import pathlib

                    print("\n🔄 Loading coffee disease model...")
                    print("📁 Path:", self.model_path)
                    print("📌 Exists:", self.model_path.exists())

                    if not self.model_path.exists():
                        raise FileNotFoundError(f"Model not found at {self.model_path}")

                    load_path = self.model_path
                    temp_dir = None
                    if load_path.suffix == ".keras" and not zipfile.is_zipfile(load_path):
                        if _looks_like_hdf5(load_path):
                            temp_dir = tempfile.TemporaryDirectory()
                            temp_path = (
                                pathlib.Path(temp_dir.name)
                                / f"{load_path.stem}.h5"
                            )
                            shutil.copyfile(load_path, temp_path)
                            load_path = temp_path
                            print(
                                "⚠️ Detected HDF5 model saved with .keras extension. "
                                "Loading as .h5 for compatibility."
                            )
                        else:
                            raise ValueError(
                                "Model file ends with .keras but is not a valid Keras zip "
                                "and does not look like an HDF5 file."
                            )

                    class LegacyDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
                        def __init__(self, *args, **kwargs):
                            kwargs.pop("groups", None)
                            super().__init__(*args, **kwargs)

                        @classmethod
                        def from_config(cls, config):
                            config.pop("groups", None)
                            return super().from_config(config)

                    # Fix for older TF versions that don't accept DepthwiseConv2D.groups
                    try:
                        model = load_model(
                            load_path,
                            compile=False,
                            custom_objects={"DepthwiseConv2D": LegacyDepthwiseConv2D},
                        )
                    finally:
                        if temp_dir is not None:
                            temp_dir.cleanup()

                    self._model = self._ensure_model_compatible(model, tf)

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
        self.used_fallback = False
        self.last_fallback_predictions: list[dict[str, float | str]] | None = None

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
            self.used_fallback = False
            self.last_fallback_predictions = None
            return predictions

        except Exception as e:
            print("❌ Primary model failed:", e)
            self._active = self._fallback
            self.notice = "Primary model failed. Using fallback ImageNet model."
            self.used_fallback = True
            fallback_predictions = self._fallback.predict(raw_bytes, top=top)
            self.last_fallback_predictions = fallback_predictions
            return fallback_predictions


# 🔥 Initialize classifier
classifier = ModelRouter(
    primary=CoffeeDiseaseClassifier(),
    fallback=VGG16Classifier(),
)
