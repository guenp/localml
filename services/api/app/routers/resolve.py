"""Server-side resource reference resolution.

Shared ``name:version`` resolution (e.g. ``assistant:v1``) for models, datasets, and prompts,
returning canonical ids.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..repositories import resolve_dataset, resolve_model_version, resolve_prompt
from ..schemas import ResolveReferenceRequest, ResolveReferenceResponse
from ..session import get_db

router = APIRouter(prefix="/resolve", tags=["resolve"])


@router.post("", response_model=ResolveReferenceResponse)
def resolve_reference(
    req: ResolveReferenceRequest, db: Session = Depends(get_db)
) -> ResolveReferenceResponse:
    if req.resource_type == "model":
        mv = resolve_model_version(db, req.reference)
        return ResolveReferenceResponse(
            resource_type=req.resource_type,
            reference=req.reference,
            id=mv.id,
            name=mv.model.name,
            version=f"v{mv.version}",
        )
    if req.resource_type == "dataset":
        ds = resolve_dataset(db, req.reference)
        return ResolveReferenceResponse(
            resource_type=req.resource_type,
            reference=req.reference,
            id=ds.id,
            name=ds.name,
            version=ds.version,
        )
    if req.resource_type == "prompt":
        prompt = resolve_prompt(db, req.reference)
        return ResolveReferenceResponse(
            resource_type=req.resource_type,
            reference=req.reference,
            id=prompt.id,
            name=prompt.name,
            version=prompt.version,
        )
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "unsupported resource type")
