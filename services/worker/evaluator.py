"""Evaluation logic.

Pluggable metric computation against a model version and dataset. The scaffold returns
placeholder values so the end-to-end flow is observable; real metric functions and a
serving/inference loop are Phase 3 work.
"""

from __future__ import annotations

import logging

log = logging.getLogger("localml.evaluator")

# Map of metric name -> placeholder value. Replace with real metric functions in Phase 3.
_PLACEHOLDER_METRICS = {
    "accuracy": 0.0,
    "exact_match": 0.0,
    "latency_p95": 0.0,
}


def run_evaluation(payload: dict) -> tuple[dict[str, float], str | None]:
    """Run an evaluation job and return (metrics, report_uri).

    Steps (real implementation, Phase 3):
      1. Resolve model artifacts + dataset from MinIO.
      2. Run predictions via the local serving runtime.
      3. Compute the requested metrics.
      4. Save an evaluation report to MinIO and return its URI.
    """
    requested = payload.get("metrics", [])
    metrics = {name: _PLACEHOLDER_METRICS.get(name, 0.0) for name in requested}
    log.info("computed placeholder metrics for job %s: %s", payload.get("job_id"), metrics)
    report_uri = None  # TODO(phase3): write report to MinIO and return its URI.
    return metrics, report_uri
