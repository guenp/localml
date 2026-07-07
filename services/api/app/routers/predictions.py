"""Prediction-job endpoints.

Creates the job record (status=queued) after resolving the model + dataset + prompt triple,
pre-flights the prompt's variables against the dataset's known columns, and enqueues the job
for the background worker. When Redis is unreachable the job runs on an in-process background
thread instead (:func:`app.prediction.schedule_inline`) so the standalone flow completes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..db import PredictionJob
from ..prediction import read_results, schedule_inline
from ..queue import enqueue_prediction
from ..repositories import (
    apply_idempotency,
    resolve_dataset,
    resolve_model_version,
    resolve_prompt,
)
from ..schemas import CreatePredictionRequest, PredictionJobResponse, PredictionResultsResponse
from ..session import get_db

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _to_response(job: PredictionJob) -> PredictionJobResponse:
    return PredictionJobResponse(
        id=job.id,
        model_version_id=job.model_version_id,
        dataset_id=job.dataset_id,
        prompt_version_id=job.prompt_version_id,
        status=job.status,
        provider=job.provider,
        config=job.config,
        completed_count=len(job.completed_examples),
        total_count=job.dataset.row_count,
        results_uri=job.results_uri,
        summary=job.summary,
        error=job.error,
    )


@router.post("", response_model=PredictionJobResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(
    req: CreatePredictionRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> PredictionJobResponse:
    mv = resolve_model_version(db, req.model)
    dataset = resolve_dataset(db, req.dataset)
    prompt = resolve_prompt(db, req.prompt)

    # Pre-flight: every prompt variable must be a column of every dataset row. Only enforceable
    # when rows were provided at registration (otherwise ``columns`` is empty — checked at run
    # time per row instead).
    if dataset.columns:
        missing = [v for v in prompt.variables if v not in dataset.columns]
        if missing:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"dataset {dataset.name}:{dataset.version} does not provide prompt "
                f"variable(s): {', '.join(missing)}",
            )

    queued_job_id: list[str] = []

    def create() -> PredictionJob:
        job = PredictionJob(
            model_version_id=mv.id,
            dataset_id=dataset.id,
            prompt_version_id=prompt.id,
            status="queued",
            provider=req.provider,
            config=req.config,
        )
        db.add(job)
        db.flush()
        if not enqueue_prediction({"job_id": job.id}):
            queued_job_id.append(job.id)
        return job

    response = apply_idempotency(
        db,
        "predictions",
        idempotency_key,
        {
            "resolved": {"model": mv.id, "dataset": dataset.id, "prompt": prompt.id},
            **req.model_dump(mode="json"),
        },
        create,
        _to_response,
    )
    # Fallback scheduling happens after apply_idempotency committed the job row, so the
    # background thread is guaranteed to see it. Idempotent replays never re-schedule
    # (``create`` doesn't run, so ``queued_job_id`` stays empty).
    for job_id in queued_job_id:
        schedule_inline(db.get_bind(), job_id)
    return response


@router.get("/{job_id}", response_model=PredictionJobResponse)
def get_prediction(job_id: str, db: Session = Depends(get_db)) -> PredictionJobResponse:
    job = db.get(PredictionJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction job not found")
    return _to_response(job)


@router.get("/{job_id}/results", response_model=PredictionResultsResponse)
def get_prediction_results(job_id: str, db: Session = Depends(get_db)) -> PredictionResultsResponse:
    job = db.get(PredictionJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction job not found")
    results = read_results(job)
    if results is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"results not available yet (job status: {job.status})",
        )
    return PredictionResultsResponse(job_id=job.id, results=results)
