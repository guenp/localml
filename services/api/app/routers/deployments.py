"""Deployment endpoints.

Validates the model version + lifecycle state, resolves the artifact, asks the serving
runtime to load it, and records an active deployment. Serving integration is stubbed in this
scaffold (Phase 4) — the endpoint URL is synthesized and ``predict`` echoes a placeholder.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..lifecycle import DEPLOYABLE
from ..schemas import CreateDeploymentRequest, DeploymentResponse
from ..store import Deployment, store

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
def create_deployment(req: CreateDeploymentRequest) -> DeploymentResponse:
    mv = store.model_versions.get(req.model_version_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    if mv.status not in DEPLOYABLE:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"model version status '{mv.status}' is not deployable; "
            f"promote to one of {sorted(DEPLOYABLE)} first",
        )
    # TODO(phase4): POST /load to the serving runtime with the resolved artifact URI.
    with store.lock:
        dep = Deployment(model_version_id=req.model_version_id, target=req.target)
        dep.endpoint_url = f"{settings.serving_url}/v1/chat/completions"
        store.deployments[dep.id] = dep
    return _to_response(dep)


@router.get("/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(deployment_id: str) -> DeploymentResponse:
    dep = store.deployments.get(deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    return _to_response(dep)


@router.delete("/{deployment_id}", status_code=status.HTTP_200_OK)
def delete_deployment(deployment_id: str) -> dict[str, str]:
    dep = store.deployments.get(deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    with store.lock:
        dep.status = "inactive"
    return {"status": "inactive"}


@router.post("/{deployment_id}/predict")
def predict(deployment_id: str, payload: dict) -> dict:
    dep = store.deployments.get(deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    # TODO(phase4): forward to the serving runtime and return its response.
    return {
        "deployment_id": deployment_id,
        "model_version_id": dep.model_version_id,
        "echo": payload,
        "note": "stubbed inference — wire up serving runtime in Phase 4",
    }
