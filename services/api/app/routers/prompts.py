"""Prompt registry endpoints.

Prompts are versioned ``str.format`` templates (see :mod:`app.templating` for the sandboxed
field grammar). Variables are auto-extracted at registration so clients and the Phase 3
prediction worker can validate inputs without parsing the template themselves; the render
endpoint applies the same strict matching (missing or extra variables → 422).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import PromptVersion
from ..repositories import apply_idempotency, get_or_create_project, resolve_prompt
from ..schemas import (
    PromptVersionResponse,
    RegisterPromptRequest,
    RenderPromptRequest,
    RenderPromptResponse,
)
from ..session import get_db
from ..templating import TemplateError, extract_variables, render

router = APIRouter(prefix="/prompts", tags=["prompts"])


def _to_response(prompt: PromptVersion) -> PromptVersionResponse:
    return PromptVersionResponse(
        id=prompt.id,
        project=prompt.project.name,
        name=prompt.name,
        version=prompt.version,
        template=prompt.template,
        variables=prompt.variables,
        metadata=prompt.meta,
    )


def _next_prompt_version(db: Session, project_id: str, name: str) -> str:
    existing = (
        db.execute(
            select(PromptVersion).where(
                PromptVersion.project_id == project_id, PromptVersion.name == name
            )
        )
        .scalars()
        .all()
    )
    return f"v{len(existing) + 1}"


@router.post("", response_model=PromptVersionResponse, status_code=status.HTTP_201_CREATED)
def register_prompt(
    req: RegisterPromptRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> PromptVersionResponse:
    try:
        variables = extract_variables(req.template)
    except TemplateError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    payload = req.model_dump(mode="json")

    def create() -> PromptVersion:
        project = get_or_create_project(db, req.project)
        version = req.version or _next_prompt_version(db, project.id, req.name)
        clash = db.execute(
            select(PromptVersion).where(
                PromptVersion.project_id == project.id,
                PromptVersion.name == req.name,
                PromptVersion.version == version,
            )
        ).scalar_one_or_none()
        if clash is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"prompt {req.name}:{version} already exists"
            )
        prompt = PromptVersion(
            project_id=project.id,
            name=req.name,
            version=version,
            template=req.template,
            variables=variables,
            meta=req.metadata,
        )
        db.add(prompt)
        db.flush()
        return prompt

    return apply_idempotency(db, "prompts", idempotency_key, payload, create, _to_response)


@router.get("/{name}", response_model=list[PromptVersionResponse])
def get_prompt(name: str, db: Session = Depends(get_db)) -> list[PromptVersionResponse]:
    prompts = (
        db.execute(
            select(PromptVersion).where(PromptVersion.name == name).order_by(PromptVersion.version)
        )
        .scalars()
        .all()
    )
    if not prompts:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prompt not found")
    return [_to_response(p) for p in prompts]


@router.get("/{name}/versions/{version}", response_model=PromptVersionResponse)
def get_prompt_version(
    name: str, version: str, db: Session = Depends(get_db)
) -> PromptVersionResponse:
    return _to_response(resolve_prompt(db, f"{name}:{version}"))


@router.post("/{name}/versions/{version}/render", response_model=RenderPromptResponse)
def render_prompt(
    name: str, version: str, req: RenderPromptRequest, db: Session = Depends(get_db)
) -> RenderPromptResponse:
    prompt = resolve_prompt(db, f"{name}:{version}")
    try:
        rendered = render(prompt.template, req.variables)
    except TemplateError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return RenderPromptResponse(name=prompt.name, version=prompt.version, rendered=rendered)
