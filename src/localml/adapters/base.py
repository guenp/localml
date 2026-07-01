"""Shared logic for framework adapters.

Adapters are **stateless**: they validate inputs, package framework-specific artifacts,
normalize metadata into the shared schema, and then call the common model-registration
path. Each framework module (``torch``, ``jax``, ``mlx``, ``huggingface``) builds on these
helpers so the platform core stays framework-neutral.
"""

from __future__ import annotations

import hashlib
import importlib
import tarfile
from pathlib import Path
from typing import Any

from .._hashing import sha256_file
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


def manifest(path: Path) -> dict[str, str]:
    """Map each file under ``path`` (relative POSIX path) to its SHA-256."""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return {p.relative_to(path).as_posix(): sha256_file(p) for p in files}


def content_digest(file_manifest: dict[str, str]) -> str:
    """Stable digest of a manifest — identifies contents independent of packaging/timestamps."""
    lines = "\n".join(f"{name}:{digest}" for name, digest in sorted(file_manifest.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def package_dir(path: Path) -> tuple[Path, str, dict[str, str]]:
    """Bundle a model directory into a ``.tar.gz`` next to it.

    Returns ``(bundle_path, content_digest, manifest)``. The digest is derived from the file
    manifest (not the gzip bytes) so it is reproducible across runs.
    """
    files = manifest(path)
    bundle = path.parent / f"{path.name}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for rel in sorted(files):
            tar.add(path / rel, arcname=rel)
    return bundle, content_digest(files), files


def framework_version(module: str) -> str | None:
    """Best-effort ``__version__`` of an optional framework, or ``None`` if not importable."""
    try:
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:
        return None


def stage_artifact(path: Path) -> str:
    """Return a ``file://`` URI for a staged artifact (directory or bundle)."""
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
