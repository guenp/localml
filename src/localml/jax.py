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

    Scaffold note: real Orbax serialization + sharding capture land in Phase 3 — until then
    only the contents of ``checkpoint_dir`` are packaged; in-memory ``params``/``state`` are
    recorded as provided but not serialized.
    """
    target = Path(checkpoint_dir) if checkpoint_dir else Path(f"./.localml/jax/{name}")
    target.mkdir(parents=True, exist_ok=True)
    # TODO(phase3): Orbax save of params/state + shape/dtype capture when jax/orbax are present.

    return base.package_and_register(
        target,
        name=name,
        framework="jax",
        meta={
            "checkpoint_format": checkpoint_format,
            "config": config or {},
            "jax_version": base.framework_version("jax"),
            # "provided", not "serialized": the bundle carries only checkpoint_dir contents
            # until the Phase 3 Orbax save lands.
            "params_provided": params is not None,
            "state_provided": state is not None,
        },
        metadata=metadata,
    )
