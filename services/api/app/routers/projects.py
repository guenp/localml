"""Project endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status

from ..repositories import apply_idempotency, get_or_create_project
from ..schemas import CreateProjectRequest, ProjectResponse
from ..store import Project, store

router = APIRouter(prefix="/projects", tags=["projects"])


def _to_response(project: Project) -> ProjectResponse:
    return ProjectResponse(id=project.id, name=project.name, description=project.description)


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    req: CreateProjectRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ProjectResponse:
    return apply_idempotency(
        "projects",
        idempotency_key,
        req.model_dump(mode="json"),
        lambda: get_or_create_project(req.name, req.description),
        _to_response,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str) -> ProjectResponse:
    project = store.projects.get(project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    return _to_response(project)
