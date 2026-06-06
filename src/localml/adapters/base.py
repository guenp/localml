"""Shared logic for framework adapters.

Adapters are **stateless**: they validate inputs, package framework-specific artifacts,
normalize metadata into the shared schema, and then call the common model-registration
path. Each framework module (``torch``, ``jax``, ``mlx``, ``huggingface``) builds on these
helpers so the platform core stays framework-neutral.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..client import get_client
from ..exceptions import ValidationError
from ..types import ModelVersion


def require_dir(model_dir: str | Path, *, required_files: list[str] | None = None) -> Path:
    """Validate that ``model_dir`` exists and contains any required files."""
    path = Path(model_dir)
    if not path.is_dir():
        raise ValidationError(f"model_dir is not a directory: {path}")
    for name in required_files or []:
        if not (path / name).exists():
            raise ValidationError(f"missing required file '{name}' in {path}")
    return path


def stage_artifact(path: Path) -> str:
    """Stage a local artifact and return its URI.

    Scaffold behavior: returns a ``file://`` URI pointing at the resolved path. Phase 2
    replaces this with a real MinIO upload (direct or pre-signed) plus checksum.
    """
    return path.resolve().as_uri()


def register(
    *,
    name: str,
    framework: str,
    artifact_uri: str,
    metadata: dict[str, Any],
) -> ModelVersion:
    """Normalize metadata and register a model version via the control plane."""
    base_meta = {"framework": framework}
    base_meta.update(metadata)
    return get_client().register_model_version(
        name=name,
        framework=framework,
        artifact_uri=artifact_uri,
        metadata=base_meta,
    )
