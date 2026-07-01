"""Model + model-version endpoints, including lifecycle promotion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import Model, ModelVersion
from ..integrations import register_mlflow_model
from ..lifecycle import InvalidTransition, can_transition
from ..repositories import (
    apply_idempotency,
    get_or_create_model,
    get_or_create_project,
    next_model_version,
)
from ..schemas import (
    ModelResponse,
    ModelVersionResponse,
    PromoteRequest,
    RegisterModelVersionRequest,
)
from ..session import get_db

router = APIRouter(prefix="/models", tags=["models"])


def _to_response(mv: ModelVersion, model_name: str) -> ModelVersionResponse:
    return ModelVersionResponse(
        id=mv.id,
        model_name=model_name,
        version=mv.version,
        framework=mv.framework,
        artifact_uri=mv.artifact_uri,
        status=mv.status,
        metadata=mv.meta,
    )


def _get_model(db: Session, model_name: str) -> Model:
    model = db.execute(select(Model).where(Model.name == model_name)).scalars().first()
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model not found")
    return model


@router.get("/{model_name}", response_model=ModelResponse)
def get_model(model_name: str, db: Session = Depends(get_db)) -> ModelResponse:
    model = _get_model(db, model_name)
    versions = (
        db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model.id)
            .order_by(ModelVersion.version)
        )
        .scalars()
        .all()
    )
    return ModelResponse(name=model_name, versions=[_to_response(v, model_name) for v in versions])


@router.post(
    "/{model_name}/versions",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_version(
    model_name: str,
    req: RegisterModelVersionRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ModelVersionResponse:
    def create() -> ModelVersion:
        project = get_or_create_project(db, req.project)
        model = get_or_create_model(db, project.id, model_name)
        mv = ModelVersion(
            model_id=model.id,
            version=next_model_version(db, model.id),
            framework=req.framework,
            artifact_uri=req.artifact_uri,
            status="created",
            meta={
                **req.metadata,
                "mlflow_model_name": register_mlflow_model(model_name) or model_name,
            },
        )
        db.add(mv)
        db.flush()
        return mv

    return apply_idempotency(
        db,
        "model_versions",
        idempotency_key,
        {"path_model_name": model_name, **req.model_dump(mode="json")},
        create,
        lambda mv: _to_response(mv, model_name),
    )


@router.get("/{model_name}/versions/{version}", response_model=ModelVersionResponse)
def get_version(
    model_name: str, version: int, db: Session = Depends(get_db)
) -> ModelVersionResponse:
    model = _get_model(db, model_name)
    mv = db.execute(
        select(ModelVersion).where(
            ModelVersion.model_id == model.id, ModelVersion.version == version
        )
    ).scalar_one_or_none()
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    return _to_response(mv, model_name)


@router.post("/{model_name}/versions/{version}/promote", response_model=ModelVersionResponse)
def promote_version(
    model_name: str, version: int, req: PromoteRequest, db: Session = Depends(get_db)
) -> ModelVersionResponse:
    model = _get_model(db, model_name)
    target = db.execute(
        select(ModelVersion).where(
            ModelVersion.model_id == model.id, ModelVersion.version == version
        )
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    if not can_transition(target.status, req.target_status):
        exc = InvalidTransition(target.status, req.target_status)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))
    target.status = req.target_status
    return _to_response(target, model_name)
