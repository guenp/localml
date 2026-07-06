"""Prediction-job execution: render prompts per dataset row, run inference, write JSONL.

This is the Phase 3 M2 loop. It normally runs on the background worker (``app.worker``);
when Redis is unavailable the API schedules it on an in-process background thread instead
(:func:`schedule_inline`) so the standalone flow still completes — never inline in a request.

Result rows are appended to ``{results_dir}/predictions/{job_id}.jsonl`` (one record per
example: input, rendered prompt, output, latency, token counts, error) and uploaded to MinIO
when available. Progress checkpoints land in ``PredictionJob.completed_examples`` after every
batch, so a re-run of the same job skips finished examples instead of re-inferring them.
Per-example failures (missing variables, backend errors) still emit a result row with
``error`` set — they never fail the job, so evaluations can score around them.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .db import Dataset, PredictionJob, PromptVersion
from .inference import InferenceProvider, get_provider
from .integrations import download_object, upload_object
from .templating import TemplateError, render

log = logging.getLogger("localml.prediction")

TERMINAL_STATUSES = frozenset({"completed", "failed"})


class PredictionError(RuntimeError):
    """The job as a whole cannot run (unreadable dataset, unknown provider, ...)."""


def results_path(job_id: str) -> Path:
    return Path(settings.results_dir) / "predictions" / f"{job_id}.jsonl"


def _results_key(job_id: str) -> str:
    return f"predictions/{job_id}.jsonl"


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise PredictionError(f"malformed JSONL: {exc}") from exc


def load_dataset_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """Load the dataset's JSONL rows: local ``artifact_uri`` first, then MinIO."""
    local = Path(dataset.artifact_uri.removeprefix("file://"))
    if local.is_file():
        return _parse_jsonl(local.read_text())
    key = f"datasets/{dataset.name}/{dataset.version}.jsonl"
    cache = Path(settings.results_dir) / key
    cache.parent.mkdir(parents=True, exist_ok=True)
    if download_object(key, str(cache)):
        return _parse_jsonl(cache.read_text())
    raise PredictionError(
        f"dataset rows unavailable: no local file at {dataset.artifact_uri!r} and no object "
        f"store copy at {key!r}"
    )


def _example_ids(dataset: Dataset, rows: list[dict[str, Any]]) -> list[str]:
    """Align rows with the stable ids recorded at registration (positional when they match)."""
    if len(rows) == len(dataset.example_ids):
        return [str(eid) for eid in dataset.example_ids]
    return [str(row.get("example_id") or f"ex-{idx:06d}") for idx, row in enumerate(rows)]


def _predict_row(
    prompt: PromptVersion,
    provider: InferenceProvider,
    config: dict[str, Any],
    example_id: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": example_id,
        "input": row,
        "rendered_prompt": None,
        "output": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }
    missing = [v for v in prompt.variables if v not in row]
    if missing:
        record["error"] = f"missing variables: {', '.join(missing)}"
        return record
    try:
        rendered = render(prompt.template, {v: row[v] for v in prompt.variables})
    except TemplateError as exc:  # defensive: template was validated at registration
        record["error"] = f"render failed: {exc}"
        return record
    record["rendered_prompt"] = rendered
    try:
        result = provider.generate(rendered, config)
    except Exception as exc:
        record["error"] = str(exc)
        return record
    record.update(
        output=result.output,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
    return record


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_prediction_job(
    session_factory: sessionmaker[Session] | Any,
    job_id: str,
    provider: InferenceProvider | None = None,
) -> None:
    """Execute one prediction job to a terminal state. Safe to re-run (resumes)."""
    db: Session = session_factory()
    try:
        job = db.get(PredictionJob, job_id)
        if job is None:
            log.warning("prediction job %s not found; skipping", job_id)
            return
        if job.status in TERMINAL_STATUSES:
            log.info("prediction job %s already %s; skipping", job_id, job.status)
            return
        job.status = "running"
        db.commit()
        _run(db, job, provider)
    except Exception as exc:
        log.exception("prediction job %s failed: %s", job_id, exc)
        db.rollback()
        job = db.get(PredictionJob, job_id)
        if job is not None:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.completed_at = datetime.now(UTC)
            db.commit()
    finally:
        db.close()


def _run(db: Session, job: PredictionJob, provider: InferenceProvider | None) -> None:
    prompt = job.prompt_version
    config = job.config or {}
    if provider is None:
        provider = get_provider(job.provider, model=job.model_version.model.name, config=config)

    rows = load_dataset_rows(job.dataset)
    ids = _example_ids(job.dataset, rows)
    done = set(job.completed_examples)
    pending = [(eid, row) for eid, row in zip(ids, rows, strict=True) if eid not in done]
    batch_size = max(1, int(config.get("batch_size", 4)))

    path = results_path(job.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # ``batch_size`` bounds in-flight concurrency, not true batching: each chunk runs through
    # a thread pool, is appended to the JSONL, and is checkpointed before the next starts.
    with path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(batch_size) as pool:
        for chunk in _chunks(pending, batch_size):
            records = list(
                pool.map(lambda item: _predict_row(prompt, provider, config, *item), chunk)
            )
            for record in records:
                fh.write(json.dumps(record) + "\n")
            fh.flush()
            # Reassign (don't mutate) so the JSON column change is tracked.
            job.completed_examples = job.completed_examples + [eid for eid, _ in chunk]
            db.commit()

    results = read_results(job) or []
    errored = sum(1 for r in results if r.get("error"))
    job.summary = {
        "total": len(results),
        "succeeded": len(results) - errored,
        "errored": errored,
        "duration_ms": (time.perf_counter() - started) * 1000,
    }
    job.results_uri = upload_object(str(path), _results_key(job.id)) or str(path)
    job.status = "completed"
    job.completed_at = datetime.now(UTC)
    db.commit()
    log.info("prediction job %s completed: %s", job.id, job.summary)


def read_results(job: PredictionJob) -> list[dict[str, Any]] | None:
    """Read a job's result records, deduplicated by ``example_id`` (last write wins).

    Prefers the deterministic local path; falls back to downloading the MinIO copy (the API
    and worker containers don't share a filesystem). Returns ``None`` when neither exists.
    """
    path = results_path(job.id)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not download_object(_results_key(job.id), str(path)):
            return None
    by_id: dict[str, dict[str, Any]] = {}
    for record in _parse_jsonl(path.read_text()):
        by_id[str(record.get("example_id"))] = record
    return list(by_id.values())


def schedule_inline(engine: Engine, job_id: str) -> None:
    """Run a job on a daemon thread against ``engine`` (the no-Redis degradation path)."""
    if not settings.inline_jobs:
        return
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    threading.Thread(
        target=run_prediction_job, args=(factory, job_id), daemon=True, name=f"predict-{job_id}"
    ).start()
