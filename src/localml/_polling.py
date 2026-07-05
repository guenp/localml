"""Shared job-handle polling.

Evaluation jobs run on the background worker and expose a ``.wait()``. This is the one place
the polling loop (exponential backoff, deadline) lives, so the Phase 3+ job handles (batch
prediction, fine-tuning) share it rather than growing their own.
"""

from __future__ import annotations

import time
from collections.abc import Callable

DEFAULT_MAX_INTERVAL = 10.0


def wait_for_terminal(
    refresh: Callable[[], object],
    current_status: Callable[[], str],
    terminal: frozenset[str],
    *,
    timeout: float,
    poll_interval: float,
    max_interval: float = DEFAULT_MAX_INTERVAL,
) -> str:
    """Poll ``refresh`` with exponential backoff until ``current_status`` is terminal.

    Returns the terminal status. Raises :class:`TimeoutError` if the deadline passes first.
    The interval doubles after each poll, capped at ``max_interval``.
    """
    deadline = time.monotonic() + timeout
    interval = poll_interval
    while current_status() not in terminal:
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout}s (status={current_status()})")
        time.sleep(interval)
        interval = min(interval * 2, max_interval)
        refresh()
    return current_status()
