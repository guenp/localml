"""Repository helpers for control-plane resources.

These helpers isolate lookup, idempotency, and ``name:version`` resolution from the HTTP
routers. The current implementation uses the process-local store; the same functions are
the handoff point for the durable SQLAlchemy implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException, status

from .store import Dataset, ModelVersion, Project, store

T = TypeVar("T")


def apply_idempotency(
    resource: str,
    key: str | None,
    payload: dict[str, Any],
    create: Callable[[], T],
    serialize: Callable[[T], Any],
) -> Any:
    """Return the first response for a matching idempotent create request."""
    if key is None:
        return serialize(create())

    request_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_key = (resource, key)
    with store.lock:
        cached = store.idempotency.get(cache_key)
        if cached is not None:
            if cached["request_hash"] != request_hash:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "idempotency key was already used with a different request body",
                )
            return cached["response"]
        created = create()
        response = serialize(created)
        store.idempotency[cache_key] = {"request_hash": request_hash, "response": response}
        return response


def get_or_create_project(name: str, description: str | None = None) -> Project:
    existing = next((p for p in store.projects.values() if p.name == name), None)
    if existing is not None:
        return existing
    project = Project(name=name, description=description)
    store.projects[project.id] = project
    return project


def resolve_model_version(ref: str) -> ModelVersion:
    """Resolve a model version id or ``name:version`` reference."""
    if ref in store.model_versions:
        return store.model_versions[ref]
    if ":" not in ref:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    name, version = _split_reference(ref)
    if not version.startswith("v"):
        version = f"v{version}"
    try:
        version_number = int(version.removeprefix("v"))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid model version") from exc
    for mv in store.models.get(name, []):
        if mv.version == version_number:
            return mv
    raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")


def resolve_dataset(ref: str) -> Dataset:
    """Resolve a dataset id or ``name:version`` reference."""
    if ref in store.dataset_versions:
        return store.dataset_versions[ref]
    if ":" not in ref:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset version not found")
    name, version = _split_reference(ref)
    for ds in store.datasets.get(name, []):
        if ds.version == version:
            return ds
    raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset version not found")


def _split_reference(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "expected a resource id or name:version reference",
        )
    name, version = ref.rsplit(":", 1)
    if not name or not version:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid name:version reference")
    return name, version
