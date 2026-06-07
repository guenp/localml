"""Run endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status

from ..integrations import create_mlflow_run, create_presigned_put_url
from ..repositories import apply_idempotency, get_or_create_project
from ..schemas import (
    CreateRunRequest as CreateRunRequest,
)
from ..schemas import (
    LogArtifactRequest,
    LogMetricsRequest,
    LogParamsRequest,
    RunResponse,
)
from ..store import Run, store

router = APIRouter(prefix="/runs", tags=["runs"])


def _to_response(run: Run) -> RunResponse:
    return RunResponse(id=run.id, project=run.project, status=run.status, config=run.config)


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    req: CreateRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunResponse:
    def create() -> Run:
        with store.lock:
            get_or_create_project(req.project)
            run = Run(
                project=req.project,
                config=req.config,
                mlflow_run_id=create_mlflow_run(req.project),
            )
            store.runs[run.id] = run
            return run

    return apply_idempotency(
        "runs", idempotency_key, req.model_dump(mode="json"), create, _to_response
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return _to_response(run)


@router.post("/{run_id}/metrics")
def log_metrics(run_id: str, req: LogMetricsRequest) -> dict[str, str]:
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    with store.lock:
        run.metrics.update(req.metrics)
        if req.status:
            run.status = req.status
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
        payload = req.model_dump()
        object_key = f"runs/{run_id}/{req.uri.rsplit('/', 1)[-1]}"
        payload["upload_url"] = create_presigned_put_url(object_key)
        run.artifacts.append(payload)
    return {"status": "ok"}
