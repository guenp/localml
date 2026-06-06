"""MLX framework adapter — ``localml.mlx``.

Captures MLX model files, tokenizer/config, quantization metadata, the MLX version, and
Apple Silicon runtime metadata.
"""

from __future__ import annotations

from typing import Any

from .adapters import base
from .types import ModelVersion


def log_model(
    name: str,
    model_dir: str,
    *,
    quantization: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelVersion:
    """Register an MLX model directory as a model version.

    Example::

        ml.mlx.log_model(name="assistant", model_dir="./mlx_model", quantization="4bit")
    """
    path = base.require_dir(model_dir)
    meta: dict[str, Any] = {"runtime": "mlx", "quantization": quantization}
    meta.update(metadata or {})
    artifact_uri = base.stage_artifact(path)
    return base.register(name=name, framework="mlx", artifact_uri=artifact_uri, metadata=meta)
