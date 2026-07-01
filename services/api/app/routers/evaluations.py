"""Evaluation endpoints.

Creates the job record (status=queued) and enqueues it for the background worker. If Redis
isn't reachable the job is still persisted and stays queued. The dataset reference and
requested metrics are held in ``config`` — this keeps the schema stable for the Phase 3 split
into separate prediction and evaluation jobs (``dataset_id`` stays nullable for now).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..db import EvaluationJob
from ..queue import enqueue_evaluation
from ..repositories import apply_idempotency, resolve_model_version
from ..schemas import CreateEvaluationRequest, EvaluationJobResponse
from ..session import get_db

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _to_response(job: EvaluationJob) -> EvaluationJobResponse:
    metrics = {m.name: m.value for m in job.metrics_rows} if job.metrics_rows else None
    return EvaluationJobResponse(
        id=job.id,
        model_version_id=job.model_version_id,
        status=job.status,
        metrics=metrics,
        report_uri=job.report_uri,
    )


@router.post("", response_model=EvaluationJobResponse, status_code=status.HTTP_201_CREATED)
def create_evaluation(
    req: CreateEvaluationRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> EvaluationJobResponse:
    mv = resolve_model_version(db, req.model_version_id)

    def create() -> EvaluationJob:
        job = EvaluationJob(
            model_version_id=mv.id,
            status="queued",
            config={
                "dataset_uri": req.dataset_uri,
                "metrics": req.metrics,
                **req.config,
            },
        )
        db.add(job)
        db.flush()
        enqueue_evaluation(
            {
                "job_id": job.id,
                "model_version_id": job.model_version_id,
                "dataset_uri": req.dataset_uri,
                "metrics": req.metrics,
                "evaluation_config": req.config,
            }
        )
        return job

    return apply_idempotency(
        db,
        "evaluations",
        idempotency_key,
        {"resolved_model_version_id": mv.id, **req.model_dump(mode="json")},
        create,
        _to_response,
    )


@router.get("/{job_id}", response_model=EvaluationJobResponse)
def get_evaluation(job_id: str, db: Session = Depends(get_db)) -> EvaluationJobResponse:
    job = db.get(EvaluationJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation job not found")
    return _to_response(job)
