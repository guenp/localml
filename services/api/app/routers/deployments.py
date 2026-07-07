"""Deployment endpoints + the OpenAI-compatible serving proxy (Phase 4).

Creating a deployment validates the model version's lifecycle state, resolves the serving
backend (see :mod:`app.serving`), health-checks it, and records the deployment — ``active``
when the backend answered, ``degraded`` when it didn't (the proxy resolves at request time,
so a backend that comes up later starts working without any update). ``PATCH`` repoints the
deployment's model version, target, or backend config — hot model swap, no process restart.

``/v1/chat/completions`` and ``/v1/completions`` forward to the backend and return its
reply; ``/predict`` is sugar that accepts ``{"prompt": ...}`` or ``{"messages": [...]}``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..db import Deployment
from ..lifecycle import DEPLOYABLE
from ..repositories import apply_idempotency, resolve_model_version
from ..schemas import CreateDeploymentRequest, DeploymentResponse, UpdateDeploymentRequest
from ..serving import check_backend_health, proxy_openai
from ..session import get_db

router = APIRouter(prefix="/deployments", tags=["deployments"])


def _to_response(dep: Deployment) -> DeploymentResponse:
    return DeploymentResponse(
        id=dep.id,
        model_version_id=dep.model_version_id,
        target=dep.target,
        status=dep.status,
        endpoint_url=dep.endpoint_url,
        config=dep.config,
    )


def _require_deployable(db: Session, reference: str) -> str:
    mv = resolve_model_version(db, reference)
    if mv.status not in DEPLOYABLE:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"model version status '{mv.status}' is not deployable; "
            f"promote to one of {sorted(DEPLOYABLE)} first",
        )
    return mv.id


@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
def create_deployment(
    req: CreateDeploymentRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DeploymentResponse:
    model_version_id = _require_deployable(db, req.model_version_id)

    def create() -> Deployment:
        dep = Deployment(
            model_version_id=model_version_id,
            target=req.target,
            status="active",
            config=req.config,
        )
        db.add(dep)
        db.flush()
        # The serving surface is the proxy, not the backend directly.
        dep.endpoint_url = f"/deployments/{dep.id}/v1/chat/completions"
        dep.status = "active" if check_backend_health(dep) else "degraded"
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


@router.patch("/{deployment_id}", response_model=DeploymentResponse)
def update_deployment(
    deployment_id: str, req: UpdateDeploymentRequest, db: Session = Depends(get_db)
) -> DeploymentResponse:
    """Hot swap: repoint the deployment's model version, target, or backend config."""
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    if req.model_version_id is not None:
        dep.model_version_id = _require_deployable(db, req.model_version_id)
    if req.target is not None:
        dep.target = req.target
    if req.config is not None:
        dep.config = {**dep.config, **req.config}
    db.flush()
    # Re-resolve against the new backend; a swap also revives an inactive deployment.
    dep.status = "active" if check_backend_health(dep) else "degraded"
    dep.updated_at = datetime.now(UTC)
    db.commit()
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


def _servable_deployment(db: Session, deployment_id: str) -> Deployment:
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    if dep.status == "inactive":
        raise HTTPException(status.HTTP_409_CONFLICT, "deployment is inactive")
    return dep


@router.post("/{deployment_id}/v1/chat/completions")
async def chat_completions(
    deployment_id: str, request: Request, db: Session = Depends(get_db)
) -> Any:
    """Proxy an OpenAI chat-completions request to the deployment's backend."""
    dep = _servable_deployment(db, deployment_id)
    return await proxy_openai(dep, "/v1/chat/completions", await request.json())


@router.post("/{deployment_id}/v1/completions")
async def completions(deployment_id: str, request: Request, db: Session = Depends(get_db)) -> Any:
    """Proxy an OpenAI text-completions request to the deployment's backend."""
    dep = _servable_deployment(db, deployment_id)
    return await proxy_openai(dep, "/v1/completions", await request.json())


@router.post("/{deployment_id}/predict")
async def predict(
    deployment_id: str, payload: dict[str, Any], db: Session = Depends(get_db)
) -> Any:
    """Non-streaming sugar over the chat proxy: accepts ``prompt`` or ``messages``."""
    dep = _servable_deployment(db, deployment_id)
    body = dict(payload)
    if "messages" not in body:
        prompt = body.pop("prompt", None)
        if prompt is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "payload must include 'messages' or 'prompt'",
            )
        body["messages"] = [{"role": "user", "content": str(prompt)}]
    body.pop("stream", None)
    return await proxy_openai(dep, "/v1/chat/completions", body)
