"""Run endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..db import Artifact, Metric, Param, Project, Run
from ..integrations import create_mlflow_run, create_presigned_put_url
from ..repositories import apply_idempotency, get_or_create_project
from ..schemas import (
    ArtifactResponse,
    CreateRunRequest,
    LogArtifactRequest,
    LogMetricsRequest,
    LogParamsRequest,
    RunResponse,
)
from ..session import get_db

router = APIRouter(prefix="/runs", tags=["runs"])

_TERMINAL_RUN_STATES = {"completed", "failed"}


def _to_response(run: Run, project_name: str) -> RunResponse:
    return RunResponse(id=run.id, project=project_name, status=run.status, config=run.config)


def _load_run(db: Session, run_id: str) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return run


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    req: CreateRunRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunResponse:
    def create() -> Run:
        project = get_or_create_project(db, req.project)
        run = Run(
            project_id=project.id,
            config=req.config,
            status="running",
            mlflow_run_id=create_mlflow_run(req.project),
        )
        db.add(run)
        db.flush()
        return run

    return apply_idempotency(
        db,
        "runs",
        idempotency_key,
        req.model_dump(mode="json"),
        create,
        lambda run: _to_response(run, req.project),
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunResponse:
    run = _load_run(db, run_id)
    project = db.get(Project, run.project_id)
    return _to_response(run, project.name if project else "")


@router.post("/{run_id}/metrics")
def log_metrics(
    run_id: str, req: LogMetricsRequest, db: Session = Depends(get_db)
) -> dict[str, str]:
    run = _load_run(db, run_id)
    for name, value in req.metrics.items():
        db.add(Metric(run_id=run.id, name=name, value=value, step=req.step))
    if req.status:
        run.status = req.status
        if req.status in _TERMINAL_RUN_STATES:
            run.completed_at = datetime.now(UTC)
    return {"status": "ok"}


@router.post("/{run_id}/params")
def log_params(run_id: str, req: LogParamsRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    run = _load_run(db, run_id)
    for name, value in req.params.items():
        db.add(Param(run_id=run.id, name=name, value=str(value)))
    return {"status": "ok"}


@router.post("/{run_id}/artifacts", response_model=ArtifactResponse)
def log_artifact(
    run_id: str, req: LogArtifactRequest, db: Session = Depends(get_db)
) -> ArtifactResponse:
    run = _load_run(db, run_id)
    object_key = f"runs/{run_id}/{req.uri.rsplit('/', 1)[-1]}"
    artifact = Artifact(
        run_id=run.id,
        uri=req.uri,
        artifact_type=req.artifact_type,
        checksum=req.checksum,
    )
    db.add(artifact)
    db.flush()
    return ArtifactResponse(
        id=artifact.id,
        uri=artifact.uri,
        artifact_type=artifact.artifact_type,
        checksum=artifact.checksum,
        upload_url=create_presigned_put_url(object_key),
    )
