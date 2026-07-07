"""Evaluation endpoints.

Phase 3 M3: evaluations are keyed on a **completed** prediction job and score its stored
JSONL results with registered metrics (see :mod:`app.evaluation`), so they can re-run without
re-inferring. The job record is created queued and enqueued for the background worker; when
Redis is unreachable it runs on an in-process background thread instead.

The legacy pre-M3 shape (``model_version_id`` + ``dataset_uri``) is still accepted but is
record-only: those jobs have no stored results to score and are never enqueued.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..db import EvaluationJob, EvaluationMetric, PredictionJob
from ..evaluation import schedule_inline, validate_metrics
from ..queue import enqueue_evaluation
from ..repositories import apply_idempotency, resolve_model_version
from ..schemas import CreateEvaluationRequest, EvaluationJobResponse
from ..session import get_db

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _to_response(job: EvaluationJob) -> EvaluationJobResponse:
    metrics = {m.name: m.value for m in job.metrics_rows} if job.metrics_rows else None
    return EvaluationJobResponse(
        id=job.id,
        prediction_job_id=job.prediction_job_id,
        model_version_id=job.model_version_id,
        status=job.status,
        metrics=metrics,
        report_uri=job.report_uri,
        error=job.error,
    )


@router.post("", response_model=EvaluationJobResponse, status_code=status.HTTP_201_CREATED)
def create_evaluation(
    req: CreateEvaluationRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> EvaluationJobResponse:
    if (req.prediction is None) == (req.model_version_id is None):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "pass exactly one of 'prediction' (a prediction-job id) or the legacy "
            "'model_version_id' + 'dataset_uri'",
        )
    if req.prediction is not None:
        return _create_for_prediction(req, db, idempotency_key)
    return _create_legacy(req, db, idempotency_key)


def _create_for_prediction(
    req: CreateEvaluationRequest, db: Session, idempotency_key: str | None
) -> EvaluationJobResponse:
    prediction = db.get(PredictionJob, req.prediction)
    if prediction is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction job not found")
    if prediction.status != "completed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"prediction job is not completed (status: {prediction.status}); "
            "evaluations score stored results",
        )
    if not req.metrics and not req.client_metrics:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "at least one metric is required"
        )
    problems = validate_metrics(req.metrics, req.config)
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "; ".join(problems))

    queued_job_id: list[str] = []

    def create() -> EvaluationJob:
        job = EvaluationJob(
            prediction_job_id=prediction.id,
            model_version_id=prediction.model_version_id,
            dataset_id=prediction.dataset_id,
            status="queued",
            config={"metrics": req.metrics, **req.config},
        )
        db.add(job)
        db.flush()
        for name, value in req.client_metrics.items():
            db.add(EvaluationMetric(evaluation_job_id=job.id, name=name, value=float(value)))
        db.flush()  # session autoflush is off; make the rows visible to the response
        if not enqueue_evaluation({"job_id": job.id}):
            queued_job_id.append(job.id)
        return job

    response = apply_idempotency(
        db,
        "evaluations",
        idempotency_key,
        {"resolved_prediction_id": prediction.id, **req.model_dump(mode="json")},
        create,
        _to_response,
    )
    # After apply_idempotency committed the row (thread guaranteed to see it); replays never
    # re-schedule (``create`` doesn't run, so ``queued_job_id`` stays empty).
    for job_id in queued_job_id:
        schedule_inline(db.get_bind(), job_id)
    return response


def _create_legacy(
    req: CreateEvaluationRequest, db: Session, idempotency_key: str | None
) -> EvaluationJobResponse:
    if req.dataset_uri is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "legacy evaluations require 'dataset_uri'"
        )
    mv = resolve_model_version(db, req.model_version_id or "")

    def create() -> EvaluationJob:
        # Record-only: there are no stored results to score, so the job is not enqueued.
        job = EvaluationJob(
            model_version_id=mv.id,
            status="queued",
            config={"dataset_uri": req.dataset_uri, "metrics": req.metrics, **req.config},
        )
        db.add(job)
        db.flush()
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
