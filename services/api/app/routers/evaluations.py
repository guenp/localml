"""Evaluation endpoints.

Creates the job record (status=queued) and enqueues it for the background worker. In this
scaffold, if Redis isn't reachable the job is created but stays queued.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..queue import enqueue_evaluation
from ..schemas import CreateEvaluationRequest, EvaluationJobResponse
from ..store import EvaluationJob, store

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _to_response(job: EvaluationJob) -> EvaluationJobResponse:
    return EvaluationJobResponse(
        id=job.id,
        model_version_id=job.model_version_id,
        status=job.status,
        metrics=job.metrics,
        report_uri=job.report_uri,
    )


@router.post("", response_model=EvaluationJobResponse, status_code=status.HTTP_201_CREATED)
def create_evaluation(req: CreateEvaluationRequest) -> EvaluationJobResponse:
    if req.model_version_id not in store.model_versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    with store.lock:
        job = EvaluationJob(
            model_version_id=req.model_version_id,
            dataset_uri=req.dataset_uri,
            requested_metrics=req.metrics,
            config=req.config,
        )
        store.evaluations[job.id] = job
    enqueue_evaluation(
        {
            "job_id": job.id,
            "model_version_id": job.model_version_id,
            "dataset_uri": job.dataset_uri,
            "metrics": job.requested_metrics,
            "evaluation_config": job.config,
        }
    )
    return _to_response(job)


@router.get("/{job_id}", response_model=EvaluationJobResponse)
def get_evaluation(job_id: str) -> EvaluationJobResponse:
    job = store.evaluations.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation job not found")
    return _to_response(job)
