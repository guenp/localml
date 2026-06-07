"""Server-side resource reference resolution."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..repositories import resolve_dataset, resolve_model_version
from ..schemas import ResolveReferenceRequest, ResolveReferenceResponse

router = APIRouter(prefix="/resolve", tags=["resolve"])


@router.post("", response_model=ResolveReferenceResponse)
def resolve_reference(req: ResolveReferenceRequest) -> ResolveReferenceResponse:
    if req.resource_type == "model":
        mv = resolve_model_version(req.reference)
        return ResolveReferenceResponse(
            resource_type=req.resource_type,
            reference=req.reference,
            id=mv.id,
            name=mv.model_name,
            version=f"v{mv.version}",
        )
    if req.resource_type == "dataset":
        ds = resolve_dataset(req.reference)
        return ResolveReferenceResponse(
            resource_type=req.resource_type,
            reference=req.reference,
            id=ds.id,
            name=ds.name,
            version=ds.version,
        )
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "unsupported resource type")
