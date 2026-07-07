"""Evaluation-job execution: score stored prediction results with registered metrics.

This is the Phase 3 M3 loop. It runs on the background worker (``app.worker``), or on an
in-process background thread when Redis is unavailable (:func:`schedule_inline`) — never
inline in a request. Evaluations are keyed on a **completed** prediction job and score its
stored JSONL results, so they can re-run without re-inferring.

Metrics live in a pluggable registry (:func:`register_metric`); each metric maps the full
list of result records to a single float (or ``None`` when nothing is scorable — recorded in
the report as skipped, not failed). Per-metric errors are captured in the report and never
fail the job; only infrastructure problems (missing results, DB errors) do, with the
traceback stored on the job and bounded retries for transient failures.
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from . import background
from .config import settings
from .db import EvaluationJob, EvaluationMetric
from .integrations import log_mlflow_metrics, upload_object
from .prediction import TERMINAL_STATUSES, read_results

log = logging.getLogger("localml.evaluation")

MetricFn = Callable[[list[dict[str, Any]], dict[str, Any]], "float | None"]

_METRICS: dict[str, MetricFn] = {}

DEFAULT_MAX_ATTEMPTS = 3
# Seconds multiplied by the attempt number between retries (patched to 0 in tests).
_RETRY_BACKOFF = 1.0


class EvaluationError(RuntimeError):
    """The job cannot be scored (no prediction link, prediction incomplete, missing results).

    Deterministic — raising this skips the retry loop and fails the job immediately.
    """


def register_metric(name: str, fn: MetricFn) -> None:
    """Register a metric: ``fn(records, config) -> float | None`` (None = nothing scorable)."""
    if not callable(fn):
        raise TypeError("metric must be callable")
    _METRICS[name] = fn


def metric_names() -> list[str]:
    return sorted(_METRICS)


def validate_metrics(names: list[str], config: dict[str, Any]) -> list[str]:
    """Return human-readable problems with a metric request (empty = valid)."""
    problems = [f"unknown metric: {name}" for name in names if name not in _METRICS]
    if "regex_match" in names and not config.get("pattern"):
        problems.append("regex_match requires config['pattern']")
    return problems


# -- built-in metrics ----------------------------------------------------------
#
# Records are prediction-result rows: example_id, input, rendered_prompt, output,
# latency_ms, prompt_tokens, completion_tokens, error. Quality metrics score rows that
# produced an output ("score around" errored rows — error_rate captures those).


def _outputs(records: list[dict[str, Any]]) -> list[str]:
    return [str(r["output"]) for r in records if not r.get("error") and r.get("output") is not None]


def _pairs(records: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[str, str]]:
    """(output, expected) pairs; the expected value comes from the dataset row itself."""
    field = config.get("expected_field", "expected")
    return [
        (str(r["output"]), str((r.get("input") or {})[field]))
        for r in records
        if not r.get("error") and r.get("output") is not None and field in (r.get("input") or {})
    ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 1])."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def _exact_match(records: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    return _mean([float(out.strip() == exp.strip()) for out, exp in _pairs(records, config)])


def _contains_expected(records: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    return _mean([float(exp in out) for out, exp in _pairs(records, config)])


def _regex_match(records: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    pattern = re.compile(config["pattern"])
    return _mean([float(pattern.search(out) is not None) for out in _outputs(records)])


def _format_validity(records: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    """Fraction of outputs in the expected format: ``json`` or non-empty text (default)."""
    fmt = config.get("format", "text")
    if fmt == "json":
        return _json_validity(records, config)
    return _mean([float(bool(out.strip())) for out in _outputs(records)])


def _json_validity(records: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    def valid(out: str) -> float:
        try:
            json.loads(out)
            return 1.0
        except json.JSONDecodeError:
            return 0.0

    return _mean([valid(out) for out in _outputs(records)])


def _latencies(records: list[dict[str, Any]]) -> list[float]:
    return [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]


def _error_rate(records: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    if not records:
        return None
    return sum(1 for r in records if r.get("error")) / len(records)


def _avg_tokens(records: list[dict[str, Any]], key: str) -> float | None:
    return _mean([float(r[key]) for r in records if r.get(key) is not None])


for _name, _fn in {
    "exact_match": _exact_match,
    "contains_expected": _contains_expected,
    "regex_match": _regex_match,
    "format_validity": _format_validity,
    "json_validity": _json_validity,
    "latency_p50": lambda records, config: _percentile(_latencies(records), 0.50),
    "latency_p95": lambda records, config: _percentile(_latencies(records), 0.95),
    "latency_p99": lambda records, config: _percentile(_latencies(records), 0.99),
    "error_rate": _error_rate,
    "avg_input_tokens": lambda records, config: _avg_tokens(records, "prompt_tokens"),
    "avg_output_tokens": lambda records, config: _avg_tokens(records, "completion_tokens"),
}.items():
    register_metric(_name, _fn)


# -- job execution -------------------------------------------------------------


def report_path(job_id: str) -> Path:
    return Path(settings.results_dir) / "evaluations" / f"{job_id}.json"


def _report_key(job_id: str) -> str:
    return f"evaluations/{job_id}.json"


def run_evaluation_job(session_factory: sessionmaker[Session] | Any, job_id: str) -> None:
    """Execute one evaluation job to a terminal state, with bounded retries."""
    db: Session = session_factory()
    try:
        job = db.get(EvaluationJob, job_id)
        if job is None:
            log.warning("evaluation job %s not found; skipping", job_id)
            return
        if job.status in TERMINAL_STATUSES:
            log.info("evaluation job %s already %s; skipping", job_id, job.status)
            return
        job.status = "running"
        db.commit()

        attempts = max(1, int((job.config or {}).get("max_attempts", DEFAULT_MAX_ATTEMPTS)))
        last_error: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                _run(db, job)
                return
            except EvaluationError:
                db.rollback()
                last_error = traceback.format_exc()
                break  # deterministic — retrying cannot help
            except Exception:
                db.rollback()
                last_error = traceback.format_exc()
                log.warning("evaluation job %s attempt %d/%d failed", job_id, attempt, attempts)
                if attempt < attempts:
                    time.sleep(_RETRY_BACKOFF * attempt)

        job = db.get(EvaluationJob, job_id)
        if job is not None:
            job.status = "failed"
            job.error = last_error
            job.completed_at = datetime.now(UTC)
            db.commit()
            log.error("evaluation job %s failed after retries", job_id)
    finally:
        db.close()


def _run(db: Session, job: EvaluationJob) -> None:
    prediction = job.prediction_job
    if prediction is None:
        raise EvaluationError(
            "evaluation has no prediction link (legacy record-only job); "
            "create it with a 'prediction' reference to score stored results"
        )
    if prediction.status != "completed":
        raise EvaluationError(
            f"prediction job {prediction.id} is not completed (status: {prediction.status})"
        )
    records = read_results(prediction)
    if records is None:
        raise EvaluationError(f"stored results unavailable for prediction job {prediction.id}")

    config = job.config or {}
    values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in config.get("metrics") or []:
        fn = _METRICS.get(name)
        if fn is None:  # pre-flighted at create; defensive against registry drift
            skipped[name] = "unknown metric"
            continue
        try:
            value = fn(records, config)
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
            continue
        if value is None:
            skipped[name] = "no scorable records"
        else:
            values[name] = float(value)

    # Merge with rows persisted at create (client-side metrics) or by a prior attempt.
    all_metrics = {m.name: m.value for m in job.metrics_rows}
    for name, value in values.items():
        if name not in all_metrics:
            db.add(EvaluationMetric(evaluation_job_id=job.id, name=name, value=value))
            all_metrics[name] = value

    errored = sum(1 for r in records if r.get("error"))
    report = {
        "evaluation_job_id": job.id,
        "prediction_job_id": prediction.id,
        "model_version_id": job.model_version_id,
        "dataset_id": job.dataset_id,
        "metrics": all_metrics,
        "skipped": skipped,
        "counts": {"total": len(records), "succeeded": len(records) - errored, "errored": errored},
        "config": config,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path = report_path(job.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    job.report_uri = upload_object(str(path), _report_key(job.id)) or str(path)
    log_mlflow_metrics(f"eval-{job.id}", all_metrics)

    job.status = "completed"
    job.error = None
    job.completed_at = datetime.now(UTC)
    db.commit()
    log.info("evaluation job %s completed: %s", job.id, all_metrics)


def schedule_inline(engine: Engine, job_id: str) -> None:
    """Run an evaluation job on a daemon thread (the no-Redis degradation path)."""
    background.schedule_inline(engine, job_id, run_evaluation_job, "evaluate")
