"""JAX framework adapter — ``localml.jax``.

Captures PyTree parameters, training state, an Orbax checkpoint directory, shape/dtype
metadata, the JAX version, and optional sharding metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import base
from .types import ModelVersion


def log_checkpoint(
    name: str,
    *,
    params: Any | None = None,
    state: Any | None = None,
    config: dict[str, Any] | None = None,
    checkpoint_format: str = "orbax",
    checkpoint_dir: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelVersion:
    """Register a JAX checkpoint as a model version.

    Example::

        ml.jax.log_checkpoint(
            name="ranker",
            params=params,
            state=train_state,
            config=config,
            checkpoint_format="orbax",
        )

    Scaffold note: real Orbax serialization + sharding capture land in Phase 2.
    """
    target = Path(checkpoint_dir) if checkpoint_dir else Path(f"./.localml/jax/{name}")
    target.mkdir(parents=True, exist_ok=True)
    # TODO(phase3): Orbax save of params/state + shape/dtype capture when jax/orbax are present.

    bundle, checksum, files = base.package_dir(target)
    meta: dict[str, Any] = {
        "checkpoint_format": checkpoint_format,
        "config": config or {},
        "jax_version": base.framework_version("jax"),
        "has_params": params is not None,
        "has_state": state is not None,
        "checksum": checksum,
        "manifest": files,
    }
    meta.update(metadata or {})
    return base.register(
        name=name, framework="jax", artifact_uri=base.stage_artifact(bundle), metadata=meta
    )
