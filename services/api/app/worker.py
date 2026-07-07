"""Background worker: consumes prediction and evaluation jobs from Redis.

Runs from the same image/codebase as the API (``python -m app.worker`` in Compose) so it
shares the ORM, templating, providers, metrics, and config. Prediction jobs execute the
inference loop in :mod:`app.prediction`; evaluation jobs score stored prediction results in
:mod:`app.evaluation`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .config import settings
from .evaluation import run_evaluation_job
from .prediction import run_prediction_job
from .queue import EVAL_QUEUE, PREDICTION_QUEUE
from .session import SessionLocal, init_db

log = logging.getLogger("localml.worker")


def handle_prediction(payload: dict[str, Any]) -> None:
    job_id = payload.get("job_id")
    if not job_id:
        log.warning("dropping prediction payload without job_id")
        return
    log.info("picked up prediction job %s", job_id)
    run_prediction_job(SessionLocal, str(job_id))


def handle_evaluation(payload: dict[str, Any]) -> None:
    job_id = payload.get("job_id")
    if not job_id:
        log.warning("dropping evaluation payload without job_id")
        return
    log.info("picked up evaluation job %s", job_id)
    run_evaluation_job(SessionLocal, str(job_id))


_HANDLERS = {
    PREDICTION_QUEUE.encode(): handle_prediction,
    EVAL_QUEUE.encode(): handle_evaluation,
}


def main() -> None:  # pragma: no cover - exercised by the Compose stack
    logging.basicConfig(level=logging.INFO)
    log.info("worker starting; consuming %s from %s", sorted(_HANDLERS), settings.redis_url)
    if settings.database_url.startswith("sqlite"):
        init_db()

    import redis

    client = None
    while True:
        if client is None:
            try:
                client = redis.from_url(settings.redis_url)
                client.ping()
            except Exception as exc:
                log.warning("redis unavailable (%s); retrying in 5s", exc)
                client = None
                time.sleep(5)
                continue
        try:
            item = client.blpop(list(_HANDLERS), timeout=5)
        except Exception as exc:
            log.warning("redis connection lost (%s); reconnecting", exc)
            client = None
            continue
        if item is None:
            continue
        queue, raw = item
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("dropping malformed payload from %s", queue)
            continue
        _HANDLERS[queue](payload)


if __name__ == "__main__":  # pragma: no cover
    main()
