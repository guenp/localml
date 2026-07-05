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

        ml.torch.log_model(
            model=model, name="classifier", example_input=batch, metadata={"architecture": "resnet"}
        )

    Scaffold note: input/output schema inference lands in Phase 3.
    """
    target = Path(save_dir) if save_dir else Path(f"./.localml/torch/{name}")
    target.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:  # optional dependency absent: package what's already there
        pass
    else:  # torch is present — a failed save must not silently register a weightless bundle
        torch.save(model.state_dict(), target / "model.pt")

    return base.package_and_register(
        target,
        name=name,
        framework="pytorch",
        meta={
            "task": None,
            "torch_version": base.framework_version("torch"),
            "has_example_input": example_input is not None,
        },
        metadata=metadata,
    )
