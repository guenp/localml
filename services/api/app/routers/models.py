"""Model + model-version endpoints, including lifecycle promotion."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..lifecycle import InvalidTransition, can_transition
from ..schemas import (
    ModelResponse,
    ModelVersionResponse,
    PromoteRequest,
    RegisterModelVersionRequest,
)
from ..store import ModelVersion, store

router = APIRouter(prefix="/models", tags=["models"])


def _to_response(mv: ModelVersion) -> ModelVersionResponse:
    return ModelVersionResponse(
        id=mv.id,
        model_name=mv.model_name,
        version=mv.version,
        framework=mv.framework,
        artifact_uri=mv.artifact_uri,
        status=mv.status,
        metadata=mv.metadata,
    )


@router.get("/{model_name}", response_model=ModelResponse)
def get_model(model_name: str) -> ModelResponse:
    versions = store.models.get(model_name)
    if not versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model not found")
    return ModelResponse(name=model_name, versions=[_to_response(v) for v in versions])


@router.post(
    "/{model_name}/versions",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_version(model_name: str, req: RegisterModelVersionRequest) -> ModelVersionResponse:
    with store.lock:
        mv = ModelVersion(
            model_name=model_name,
            version=store.next_version(model_name),
            framework=req.framework,
            artifact_uri=req.artifact_uri,
            metadata=req.metadata,
        )
        store.models.setdefault(model_name, []).append(mv)
        store.model_versions[mv.id] = mv
    # TODO(phase1): register MLflow model version; persist to Postgres; audit event.
    return _to_response(mv)


@router.get("/{model_name}/versions/{version}", response_model=ModelVersionResponse)
def get_version(model_name: str, version: int) -> ModelVersionResponse:
    for mv in store.models.get(model_name, []):
        if mv.version == version:
            return _to_response(mv)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")


@router.post("/{model_name}/versions/{version}/promote", response_model=ModelVersionResponse)
def promote_version(model_name: str, version: int, req: PromoteRequest) -> ModelVersionResponse:
    target = next((mv for mv in store.models.get(model_name, []) if mv.version == version), None)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    if not can_transition(target.status, req.target_status):
        exc = InvalidTransition(target.status, req.target_status)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    with store.lock:
        target.status = req.target_status
    return _to_response(target)
