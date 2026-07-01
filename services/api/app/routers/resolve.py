"""Server-side resource reference resolution.

Shared ``name:version`` resolution (e.g. ``assistant:v1``) for models and datasets, returning
canonical ids. Prompts join this in Phase 3.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..repositories import resolve_dataset, resolve_model_version
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
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "unsupported resource type")
