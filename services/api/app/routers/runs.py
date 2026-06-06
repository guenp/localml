"""Run endpoints.

Creates the platform run record. In Phase 1 this also creates a backing MLflow tracking run
and persists to Postgres; here we keep it in-memory and auto-create the project if needed.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    CreateRunRequest as CreateRunRequest,
)
from ..schemas import (
    LogArtifactRequest,
    LogMetricsRequest,
    LogParamsRequest,
    RunResponse,
)
from ..store import Project, Run, store

router = APIRouter(prefix="/runs", tags=["runs"])


def _ensure_project(name: str) -> None:
    if not any(p.name == name for p in store.projects.values()):
        project = Project(name=name)
        store.projects[project.id] = project


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(req: CreateRunRequest) -> RunResponse:
    with store.lock:
        _ensure_project(req.project)
        run = Run(project=req.project, config=req.config)
        store.runs[run.id] = run
    # TODO(phase1): create MLflow run, persist to Postgres, capture mlflow_run_id.
    return RunResponse(id=run.id, project=run.project, status=run.status, config=run.config)


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return RunResponse(id=run.id, project=run.project, status=run.status, config=run.config)


@router.post("/{run_id}/metrics")
def log_metrics(run_id: str, req: LogMetricsRequest) -> dict[str, str]:
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    with store.lock:
        run.metrics.update(req.metrics)
        if req.status:
            run.status = req.status
    # TODO(phase1): mlflow.log_metric for each; update run summary in Postgres.
    return {"status": "ok"}


@router.post("/{run_id}/params")
def log_params(run_id: str, req: LogParamsRequest) -> dict[str, str]:
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    with store.lock:
        run.params.update(req.params)
    return {"status": "ok"}


@router.post("/{run_id}/artifacts")
def log_artifact(run_id: str, req: LogArtifactRequest) -> dict[str, str]:
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    with store.lock:
        run.artifacts.append(req.model_dump())
    # TODO(phase2): accept upload / issue pre-signed MinIO URL; record checksum.
    return {"status": "ok"}
