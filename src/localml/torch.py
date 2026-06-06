"""PyTorch framework adapter — ``localml.torch``.

Captures ``state_dict``, model config, optional input/output schema, the PyTorch version,
and Python dependencies, then registers the result as a shared model version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import base
from .types import ModelVersion


def log_model(
    model: Any,
    name: str,
    *,
    example_input: Any | None = None,
    metadata: dict[str, Any] | None = None,
    save_dir: str | None = None,
) -> ModelVersion:
    """Serialize a PyTorch model and register it as a model version.

    Example::

        ml.torch.log_model(model=model, name="classifier", example_input=batch,
                           metadata={"architecture": "resnet"})

    Scaffold note: real ``state_dict`` serialization and schema inference land in Phase 2.
    """
    meta: dict[str, Any] = {"task": None}
    meta.update(metadata or {})
    if example_input is not None:
        meta["has_example_input"] = True

    target = Path(save_dir) if save_dir else Path(f"./.localml/torch/{name}")
    target.mkdir(parents=True, exist_ok=True)
    # TODO(phase2): torch.save(model.state_dict(), target / "model.pt"); capture versions.
    artifact_uri = base.stage_artifact(target)

    return base.register(name=name, framework="pytorch", artifact_uri=artifact_uri, metadata=meta)
