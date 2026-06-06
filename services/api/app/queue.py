"""Job queue integration.

Enqueues evaluation jobs onto Redis for the background worker. If Redis is unavailable
(e.g. running the API standalone for tests), it degrades to a no-op so the API stays
usable; the worker simply has nothing to pick up.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import settings

log = logging.getLogger("localml.queue")

EVAL_QUEUE = "localml:evaluations"

try:  # redis is optional at import time
    import redis

    _redis: Any | None = redis.from_url(settings.redis_url)
except Exception:  # pragma: no cover - optional dependency / unreachable broker
    _redis = None


def enqueue_evaluation(payload: dict[str, Any]) -> bool:
    """Push an evaluation job payload onto the queue. Returns False if unavailable."""
    if _redis is None:
        log.warning("redis unavailable; evaluation %s not enqueued", payload.get("job_id"))
        return False
    try:
        _redis.rpush(EVAL_QUEUE, json.dumps(payload))
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("failed to enqueue evaluation: %s", exc)
        return False
