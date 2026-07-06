"""FastAPI control plane entrypoint.

Coordinates platform metadata, lifecycle, jobs, and serving. See docs/design.md §4.3.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from .auth import require_auth
from .config import settings
from .routers import datasets, deployments, evaluations, models, projects, prompts, resolve, runs
from .session import init_db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # SQLite (local/dev) has no migration step; create tables on boot. Postgres uses Alembic.
    if settings.database_url.startswith("sqlite"):
        init_db()
    yield


app = FastAPI(
    title="localml control plane",
    version="0.1.0",
    description="Local ML experimentation platform control plane.",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


# All resource routers require auth (a no-op when LOCALML_AUTH_BYPASS=true).
_protected = [Depends(require_auth)]
app.include_router(projects.router, dependencies=_protected)
app.include_router(runs.router, dependencies=_protected)
app.include_router(models.router, dependencies=_protected)
app.include_router(datasets.router, dependencies=_protected)
app.include_router(prompts.router, dependencies=_protected)
app.include_router(evaluations.router, dependencies=_protected)
app.include_router(deployments.router, dependencies=_protected)
app.include_router(resolve.router, dependencies=_protected)
