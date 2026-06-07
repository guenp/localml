"""Dataset registry endpoints."""

from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Header, HTTPException, status

from ..repositories import apply_idempotency, get_or_create_project, resolve_dataset
from ..schemas import DatasetResponse, RegisterDatasetRequest
from ..store import Dataset, store

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


def _to_response(dataset: Dataset) -> DatasetResponse:
    return DatasetResponse(
        id=dataset.id,
        project=dataset.project,
        name=dataset.name,
        version=dataset.version,
        artifact_uri=dataset.artifact_uri,
        row_count=dataset.row_count,
        example_ids=dataset.example_ids,
        metadata=dataset.metadata,
    )


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def register_dataset(
    req: RegisterDatasetRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DatasetResponse:
    payload = req.model_dump(mode="json")

    def create() -> Dataset:
        rows = req.rows or []
        example_ids = _stable_example_ids(rows)
        with store.lock:
            get_or_create_project(req.project)
            version = req.version or store.next_dataset_version(req.name)
            if any(ds.version == version for ds in store.datasets.get(req.name, [])):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"dataset {req.name}:{version} already exists",
                )
            dataset = Dataset(
                project=req.project,
                name=req.name,
                version=version,
                artifact_uri=req.artifact_uri,
                row_count=len(rows),
                example_ids=example_ids,
                metadata=req.metadata,
            )
            store.datasets.setdefault(dataset.name, []).append(dataset)
            store.dataset_versions[dataset.id] = dataset
            return dataset

    return apply_idempotency("datasets", idempotency_key, payload, create, _to_response)


@router.get("/{name}", response_model=list[DatasetResponse])
def get_dataset(name: str) -> list[DatasetResponse]:
    datasets = store.datasets.get(name)
    if not datasets:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    return [_to_response(ds) for ds in datasets]


@router.get("/{name}/versions/{version}", response_model=DatasetResponse)
def get_dataset_version(name: str, version: str) -> DatasetResponse:
    return _to_response(resolve_dataset(f"{name}:{version}"))
