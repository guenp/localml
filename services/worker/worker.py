"""Background evaluation worker.

Dequeues evaluation jobs from Redis, resolves model + dataset artifacts, runs the
evaluation loop, computes metrics, writes a report to MinIO, logs metrics to MLflow, and
updates job status in Postgres.

This scaffold implements the loop structure and a stub evaluator. The artifact/serving/DB
integrations are marked with TODOs and land in Phase 3 of the roadmap.
"""

from __future__ import annotations

import json
import logging
import os
import time

from evaluator import run_evaluation

log = logging.getLogger("localml.worker")
logging.basicConfig(level=logging.INFO)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
EVAL_QUEUE = "localml:evaluations"


def _connect():  # type: ignore[no-untyped-def]
    import redis

    return redis.from_url(REDIS_URL)


def handle(payload: dict) -> None:
    job_id = payload.get("job_id")
    log.info("picked up evaluation job %s", job_id)
    # TODO(phase3): mark status=running in Postgres; resolve artifacts from MinIO.
    try:
        metrics, _report_uri = run_evaluation(payload)
        log.info("job %s completed: %s", job_id, metrics)
        # TODO(phase3): log metrics to MLflow; update status=completed + report_uri in DB.
    except Exception as exc:
        log.exception("job %s failed: %s", job_id, exc)
        # TODO(phase3): mark status=failed with traceback summary; bounded retry.


def main() -> None:
    log.info("worker starting; consuming %s from %s", EVAL_QUEUE, REDIS_URL)
    try:
        client = _connect()
    except Exception as exc:
        log.error("could not connect to redis (%s); idling", exc)
        while True:
            time.sleep(5)

    while True:
        item = client.blpop(EVAL_QUEUE, timeout=5)
        if item is None:
            continue
        _, raw = item
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("dropping malformed job payload")
            continue
        handle(payload)


if __name__ == "__main__":
    main()
