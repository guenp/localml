"""Run lifecycle context manager."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ._state import set_current_run
from .client import get_client
from .types import Run


@contextmanager
def start_run(project: str, config: dict[str, Any] | None = None) -> Iterator[Run]:
    """Start a tracked run and set it as the active run for the duration of the block.

    On a clean exit the run is marked ``completed``; on an exception it is marked
    ``failed`` and the exception propagates.

    Example::

        with localml.start_run(project="demo", config={"lr": 1e-3}) as run:
            localml.log_metrics({"accuracy": 0.91})
    """
    client = get_client()
    run = client.create_run(project=project, config=config or {})
    set_current_run(run)
    try:
        yield run
    except Exception:
        run.status = "failed"
        client.complete_run(run.id, status="failed")
        raise
    else:
        run.status = "completed"
        client.complete_run(run.id, status="completed")
    finally:
        set_current_run(None)
