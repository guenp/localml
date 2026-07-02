"""Repository helpers for control-plane resources.

These isolate cross-cutting concerns — idempotency, get-or-create, and ``name:version``
resolution — from the HTTP routers, operating on a SQLAlchemy :class:`~sqlalchemy.orm.Session`.
Straightforward CRUD lives inline in the routers; anything shared or subtle lives here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import Dataset, IdempotencyKey, Model, ModelVersion, Project, User

T = TypeVar("T")

DEFAULT_USERNAME = "local"
DEFAULT_PROJECT = "local"


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _as_json(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def apply_idempotency(
    db: Session,
    resource: str,
    key: str | None,
    payload: dict[str, Any],
    create: Callable[[], T],
    serialize: Callable[[T], Any],
) -> Any:
    """Return the first response for a matching idempotent create request.

    A replay with the same key and body returns the stored response; the same key with a
    different body is a 409. Without a key, every call creates.

    Commits before returning so the write is durable before the HTTP response is sent (the
    ``get_db`` teardown commit can otherwise run after the response, breaking read-your-writes
    for a client's immediate follow-up request).
    """
    if key is None:
        response = serialize(create())
        db.commit()
        return response

    request_hash = _hash_payload(payload)
    existing = db.execute(
        select(IdempotencyKey).where(IdempotencyKey.resource == resource, IdempotencyKey.key == key)
    ).scalar_one_or_none()
    if existing is not None:
        if existing.request_hash != request_hash:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "idempotency key was already used with a different request body",
            )
        return existing.response

    response = serialize(create())
    db.add(
        IdempotencyKey(
            resource=resource, key=key, request_hash=request_hash, response=_as_json(response)
        )
    )
    try:
        db.commit()
    except IntegrityError:  # concurrent replay raced us; return the stored response
        db.rollback()
        stored = db.execute(
            select(IdempotencyKey).where(
                IdempotencyKey.resource == resource, IdempotencyKey.key == key
            )
        ).scalar_one()
        return stored.response
    return response


def get_or_create_user(db: Session, username: str = DEFAULT_USERNAME) -> User:
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None:
        user = User(username=username, display_name=username)
        db.add(user)
        db.flush()
    return user


def get_or_create_project(db: Session, name: str, description: str | None = None) -> Project:
    project = db.execute(select(Project).where(Project.name == name)).scalar_one_or_none()
    if project is not None:
        return project
    project = Project(name=name, description=description, owner_user_id=get_or_create_user(db).id)
    db.add(project)
    db.flush()
    return project


def get_or_create_model(db: Session, project_id: str, name: str) -> Model:
    model = db.execute(
        select(Model).where(Model.project_id == project_id, Model.name == name)
    ).scalar_one_or_none()
    if model is None:
        model = Model(project_id=project_id, name=name)
        db.add(model)
        db.flush()
    return model


def next_model_version(db: Session, model_id: str) -> int:
    versions = (
        db.execute(select(ModelVersion).where(ModelVersion.model_id == model_id)).scalars().all()
    )
    return len(versions) + 1


def resolve_model_version(db: Session, ref: str) -> ModelVersion:
    """Resolve a model-version id or ``name:version`` reference (e.g. ``assistant:v1``)."""
    by_id = db.get(ModelVersion, ref)
    if by_id is not None:
        return by_id
    name, version = _split_reference(ref, "model version")
    try:
        version_number = int(version.removeprefix("v"))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid model version") from exc
    mv = db.execute(
        select(ModelVersion)
        .join(Model, ModelVersion.model_id == Model.id)
        .where(Model.name == name, ModelVersion.version == version_number)
    ).scalar_one_or_none()
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    return mv


def resolve_dataset(db: Session, ref: str) -> Dataset:
    """Resolve a dataset id or ``name:version`` reference."""
    by_id = db.get(Dataset, ref)
    if by_id is not None:
        return by_id
    name, version = _split_reference(ref, "dataset")
    ds = db.execute(
        select(Dataset).where(Dataset.name == name, Dataset.version == version)
    ).scalar_one_or_none()
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset version not found")
    return ds


def _split_reference(ref: str, resource: str) -> tuple[str, str]:
    if ":" not in ref:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{resource} not found")
    name, version = ref.rsplit(":", 1)
    if not name or not version:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid name:version reference")
    return name, version
