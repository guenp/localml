"""Dataset registry helpers."""

from __future__ import annotations

from typing import Any

from .client import get_client
from .types import Dataset


def register(
    *,
    project: str,
    name: str,
    artifact_uri: str,
    rows: list[dict[str, Any]] | None = None,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Dataset:
    """Register a JSONL dataset and return the created version."""
    return get_client().register_dataset(
        project=project,
        name=name,
        artifact_uri=artifact_uri,
        rows=rows,
        version=version,
        metadata=metadata,
    )


def get(name: str) -> list[Dataset]:
    """Return all registered versions for a dataset name."""
    return get_client().get_datasets(name)
