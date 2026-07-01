"""Dataset registry endpoints.

Datasets are versioned JSONL collections. Each row gets a **stable** ``example_id`` (either
the caller's own or a content hash) so predictions and evaluations can be aligned per-row and
compared across variants later (Phase 3). The row payloads themselves live in MinIO; the
control plane stores metadata plus the ordered ``example_id`` list.
"""

from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import Dataset
from ..integrations import create_presigned_put_url
from ..repositories import apply_idempotency, get_or_create_project, resolve_dataset
from ..schemas import DatasetResponse, RegisterDatasetRequest
from ..session import get_db

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _stable_example_ids(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for idx, row in enumerate(rows):
        raw = row.get("example_id")
        if raw is not None:
            ids.append(str(raw))
            continue
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        ids.append(f"ex-{idx:06d}-{digest}")
    if len(ids) != len(set(ids)):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "example_id values must be unique"
        )
    return ids


def _to_response(dataset: Dataset, upload_url: str | None = None) -> DatasetResponse:
    return DatasetResponse(
        id=dataset.id,
        project=dataset.project.name,
        name=dataset.name,
        version=dataset.version,
        artifact_uri=dataset.artifact_uri,
        row_count=dataset.row_count,
        example_ids=dataset.example_ids,
        metadata=dataset.meta,
        upload_url=upload_url,
    )


def _next_dataset_version(db: Session, project_id: str, name: str) -> str:
    existing = (
        db.execute(select(Dataset).where(Dataset.project_id == project_id, Dataset.name == name))
        .scalars()
        .all()
    )
    return f"v{len(existing) + 1}"


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def register_dataset(
    req: RegisterDatasetRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DatasetResponse:
    payload = req.model_dump(mode="json")

    def create() -> DatasetResponse:
        rows = req.rows or []
        example_ids = _stable_example_ids(rows)
        project = get_or_create_project(db, req.project)
        version = req.version or _next_dataset_version(db, project.id, req.name)
        clash = db.execute(
            select(Dataset).where(
                Dataset.project_id == project.id,
                Dataset.name == req.name,
                Dataset.version == version,
            )
        ).scalar_one_or_none()
        if clash is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"dataset {req.name}:{version} already exists"
            )
        dataset = Dataset(
            project_id=project.id,
            name=req.name,
            version=version,
            artifact_uri=req.artifact_uri,
            row_count=len(rows),
            example_ids=example_ids,
            meta=req.metadata,
        )
        db.add(dataset)
        db.flush()
        upload_url = create_presigned_put_url(f"datasets/{req.name}/{version}.jsonl")
        return _to_response(dataset, upload_url)

    # ``create`` already serializes (it needs the presigned URL, which is not persisted).
    return apply_idempotency(db, "datasets", idempotency_key, payload, create, lambda r: r)


@router.get("/{name}", response_model=list[DatasetResponse])
def get_dataset(name: str, db: Session = Depends(get_db)) -> list[DatasetResponse]:
    datasets = (
        db.execute(select(Dataset).where(Dataset.name == name).order_by(Dataset.version))
        .scalars()
        .all()
    )
    if not datasets:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    return [_to_response(ds) for ds in datasets]


@router.get("/{name}/versions/{version}", response_model=DatasetResponse)
def get_dataset_version(name: str, version: str, db: Session = Depends(get_db)) -> DatasetResponse:
    return _to_response(resolve_dataset(db, f"{name}:{version}"))
