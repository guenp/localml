"""FastAPI control plane entrypoint.

Coordinates platform metadata, lifecycle, jobs, and serving. See docs/design.md §4.3.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI

from .auth import require_auth
from .routers import deployments, evaluations, models, projects, runs

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="localml control plane",
    version="0.1.0",
    description="Local ML experimentation platform control plane (scaffold).",
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


# All resource routers require auth (a no-op when LOCALML_AUTH_BYPASS=true).
_protected = [Depends(require_auth)]
app.include_router(projects.router, dependencies=_protected)
app.include_router(runs.router, dependencies=_protected)
app.include_router(models.router, dependencies=_protected)
app.include_router(evaluations.router, dependencies=_protected)
app.include_router(deployments.router, dependencies=_protected)
