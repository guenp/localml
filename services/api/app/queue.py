"""Job queue integration.

Enqueues evaluation and prediction jobs onto Redis for the background worker. If Redis is
unavailable (e.g. running the API standalone for tests), it degrades to a no-op so the API
stays usable; prediction jobs then fall back to an in-process background thread (see
:func:`app.prediction.schedule_inline`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import settings

log = logging.getLogger("localml.queue")

EVAL_QUEUE = "localml:evaluations"
PREDICTION_QUEUE = "localml:predictions"

try:  # redis is optional at import time
    import redis

    _redis: Any | None = redis.from_url(settings.redis_url)
except Exception:  # pragma: no cover - optional dependency / unreachable broker
    _redis = None


def _enqueue(queue: str, payload: dict[str, Any]) -> bool:
    if _redis is None:
        log.warning("redis unavailable; job %s not enqueued", payload.get("job_id"))
        return False
    try:
        _redis.rpush(queue, json.dumps(payload))
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("failed to enqueue onto %s: %s", queue, exc)
        return False


def enqueue_evaluation(payload: dict[str, Any]) -> bool:
    """Push an evaluation job payload onto the queue. Returns False if unavailable."""
    return _enqueue(EVAL_QUEUE, payload)


def enqueue_prediction(payload: dict[str, Any]) -> bool:
    """Push a prediction job payload onto the queue. Returns False if unavailable."""
    return _enqueue(PREDICTION_QUEUE, payload)
