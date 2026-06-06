"""Project endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import CreateProjectRequest, ProjectResponse
from ..store import Project, store

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(req: CreateProjectRequest) -> ProjectResponse:
    with store.lock:
        existing = next((p for p in store.projects.values() if p.name == req.name), None)
        if existing:
            return ProjectResponse(
                id=existing.id, name=existing.name, description=existing.description
            )
        project = Project(name=req.name, description=req.description)
        store.projects[project.id] = project
    return ProjectResponse(id=project.id, name=project.name, description=project.description)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str) -> ProjectResponse:
    project = store.projects.get(project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    return ProjectResponse(id=project.id, name=project.name, description=project.description)
