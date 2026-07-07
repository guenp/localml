"""Comparison endpoint: two prediction/evaluation jobs across aligned example ids."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..comparison import build_comparison
from ..schemas import ComparisonResponse
from ..session import get_db

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("", response_model=ComparisonResponse)
def compare_jobs(
    a: str = Query(description="Prediction- or evaluation-job id (variant A)"),
    b: str = Query(description="Prediction- or evaluation-job id (variant B)"),
    max_examples: int = Query(20, ge=0, le=500, description="Cap on changed examples returned"),
    db: Session = Depends(get_db),
) -> ComparisonResponse:
    return ComparisonResponse(**build_comparison(db, a, b, max_examples))
