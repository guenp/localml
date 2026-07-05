"""Hugging Face framework adapter — ``localml.huggingface``.

Captures ``config.json``, tokenizer files, safetensors/bin weights, generation config, and
Hugging Face model metadata.
"""

from __future__ import annotations

from typing import Any

from .adapters import base
from .types import ModelVersion


def log_pretrained(
    name: str,
    model_dir: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ModelVersion:
    """Register a Hugging Face pretrained model directory as a model version.

    Example::

        ml.huggingface.log_pretrained(name="hf-assistant", model_dir="./model")
    """
    path = base.require_dir(model_dir, required_files=["config.json"])
    return base.package_and_register(
        path,
        name=name,
        framework="huggingface",
        meta={
            "source": "huggingface",
            "transformers_version": base.framework_version("transformers"),
        },
        metadata=metadata,
    )
