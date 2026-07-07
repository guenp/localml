"""In-process background execution for queued jobs (the no-Redis degradation path).

Jobs normally run on the worker (``app.worker``) via Redis. When the queue is unavailable,
routers schedule the job runner on a daemon thread instead — never inline in a request — so
the standalone (SQLite, no-Docker) flow still completes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .config import settings


def schedule_inline(
    engine: Engine, job_id: str, runner: Callable[[Any, str], None], label: str
) -> None:
    """Run ``runner(session_factory, job_id)`` on a daemon thread against ``engine``."""
    if not settings.inline_jobs:
        return
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    threading.Thread(
        target=runner, args=(factory, job_id), daemon=True, name=f"{label}-{job_id}"
    ).start()
