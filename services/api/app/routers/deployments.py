"""Deployment endpoints.

Validates the model version + lifecycle state, resolves the artifact, asks the serving
runtime to load it, and records an active deployment. The serving runtime is an
OpenAI-compatible proxy (Ollama / MLX-LM / llama.cpp — all expose ``/v1/chat/completions``);
the load + proxy wiring lands in Phase 4, so ``predict`` currently echoes a placeholder.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..config import settings
from ..db import Deployment, ModelVersion
from ..lifecycle import DEPLOYABLE
from ..repositories import apply_idempotency
from ..schemas import CreateDeploymentRequest, DeploymentResponse
from ..session import get_db

router = APIRouter(prefix="/deployments", tags=["deployments"])


def _to_response(dep: Deployment) -> DeploymentResponse:
    return DeploymentResponse(
        id=dep.id,
        model_version_id=dep.model_version_id,
        target=dep.target,
        status=dep.status,
        endpoint_url=dep.endpoint_url,
    )


@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
def create_deployment(
    req: CreateDeploymentRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DeploymentResponse:
    mv = db.get(ModelVersion, req.model_version_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    if mv.status not in DEPLOYABLE:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"model version status '{mv.status}' is not deployable; "
            f"promote to one of {sorted(DEPLOYABLE)} first",
        )

    def create() -> Deployment:
        # TODO(phase4): POST the resolved artifact to the serving runtime's load endpoint.
        dep = Deployment(
            model_version_id=req.model_version_id,
            target=req.target,
            status="active",
            endpoint_url=f"{settings.serving_url}/v1/chat/completions",
        )
        db.add(dep)
        db.flush()
        return dep

    return apply_idempotency(
        db,
        "deployments",
        idempotency_key,
        req.model_dump(mode="json"),
        create,
        _to_response,
    )


@router.get("/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(deployment_id: str, db: Session = Depends(get_db)) -> DeploymentResponse:
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    return _to_response(dep)


@router.delete("/{deployment_id}", status_code=status.HTTP_200_OK)
def delete_deployment(deployment_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    dep.status = "inactive"
    dep.updated_at = datetime.now(UTC)
    db.commit()
    return {"status": "inactive"}


@router.post("/{deployment_id}/predict")
def predict(deployment_id: str, payload: dict, db: Session = Depends(get_db)) -> dict:
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    # TODO(phase4): proxy to the OpenAI-compatible serving runtime and return its response.
    return {
        "deployment_id": deployment_id,
        "model_version_id": dep.model_version_id,
        "echo": payload,
        "note": "stubbed inference — Phase 4 wires up the OpenAI-compatible serving proxy",
    }
