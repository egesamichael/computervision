#!/usr/bin/env python3
"""
Convert a legacy Teachable Machine .h5 model to a modern .keras file.

Usage:
  python scripts/convert_teachable_model.py \
    --in models/keras_model.h5 \
    --out models/coffee_disease_compatible.keras

Run this in a Python 3.10/3.11 environment with TensorFlow 2.13.x.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_model(path: Path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    class LegacyDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            config.pop("groups", None)
            return super().from_config(config)

    return load_model(
        path,
        compile=False,
        custom_objects={"DepthwiseConv2D": LegacyDepthwiseConv2D},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path}")
    if output_path.suffix != ".keras":
        raise SystemExit("Output path must end with .keras")

    print(f"Loading legacy model from {input_path}...")
    model = _load_model(input_path)
    print("Saving converted model...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Converted model saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
