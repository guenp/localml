"""Context-local tracking of the active run.

Kept separate from the public API so adapters and ops modules can discover the current
run without import cycles. Uses :class:`contextvars.ContextVar` so concurrent runs (e.g.
in async or threaded notebooks) don't clobber each other.
"""

from __future__ import annotations

from contextvars import ContextVar

from .types import Run

_current_run: ContextVar[Run | None] = ContextVar("localml_current_run", default=None)


def set_current_run(run: Run | None) -> None:
    _current_run.set(run)


def get_current_run() -> Run:
    run = _current_run.get()
    if run is None:
        raise RuntimeError(
            "No active run. Use `with localml.start_run(...) as run:` before logging."
        )
    return run


def maybe_current_run() -> Run | None:
    return _current_run.get()
