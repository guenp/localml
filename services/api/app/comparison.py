"""Comparison reports: two prediction/evaluation jobs, aligned on stable ``example_id``s.

Phase 3 M4. A reference may be a prediction-job id or an evaluation-job id (an evaluation
compares through the prediction it scored); both underlying predictions must be completed
with readable results. The report says what varied between the variants (model / prompt /
dataset / provider / config), how the aligned rows agree, and — when both references are
evaluations — the per-metric deltas.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .db import EvaluationJob, PredictionJob
from .prediction import read_results

_VARIANT_FIELDS = (
    ("model_version", "model_version_id"),
    ("prompt_version", "prompt_version_id"),
    ("dataset", "dataset_id"),
    ("provider", "provider"),
    ("config", "config"),
)


def _resolve(db: Session, ref: str) -> tuple[PredictionJob, EvaluationJob | None]:
    """Resolve a job reference to its prediction (and evaluation, when ``ref`` is one)."""
    prediction = db.get(PredictionJob, ref)
    if prediction is not None:
        return prediction, None
    evaluation = db.get(EvaluationJob, ref)
    if evaluation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no prediction or evaluation job {ref!r}")
    if evaluation.prediction_job is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"evaluation job {ref!r} has no prediction link (legacy record-only job)",
        )
    return evaluation.prediction_job, evaluation


def _records(prediction: PredictionJob) -> dict[str, dict[str, Any]]:
    if prediction.status != "completed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"prediction job {prediction.id} is not completed (status: {prediction.status})",
        )
    records = read_results(prediction)
    if records is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"results unavailable for prediction job {prediction.id}"
        )
    return {str(r.get("example_id")): r for r in records}


def _side(prediction: PredictionJob, evaluation: EvaluationJob | None) -> dict[str, Any]:
    return {
        "job_id": evaluation.id if evaluation is not None else prediction.id,
        "prediction_job_id": prediction.id,
        "model_version_id": prediction.model_version_id,
        "prompt_version_id": prediction.prompt_version_id,
        "dataset_id": prediction.dataset_id,
        "provider": prediction.provider,
        "metrics": (
            {m.name: m.value for m in evaluation.metrics_rows} if evaluation is not None else None
        ),
    }


def _delta(a: float | None, b: float | None) -> dict[str, float | None]:
    return {"a": a, "b": b, "delta": b - a if a is not None and b is not None else None}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_comparison(db: Session, ref_a: str, ref_b: str, max_examples: int) -> dict[str, Any]:
    pred_a, eval_a = _resolve(db, ref_a)
    pred_b, eval_b = _resolve(db, ref_b)
    rows_a = _records(pred_a)
    rows_b = _records(pred_b)

    differs = [
        name for name, attr in _VARIANT_FIELDS if getattr(pred_a, attr) != getattr(pred_b, attr)
    ]

    aligned = [eid for eid in rows_a if eid in rows_b]
    both_ok = [
        eid for eid in aligned if not rows_a[eid].get("error") and not rows_b[eid].get("error")
    ]
    agreements = [eid for eid in both_ok if rows_a[eid].get("output") == rows_b[eid].get("output")]
    changed = [eid for eid in both_ok if eid not in set(agreements)]

    metrics: dict[str, dict[str, float | None]] = {}
    if eval_a is not None and eval_b is not None:
        metrics_a = {m.name: m.value for m in eval_a.metrics_rows}
        metrics_b = {m.name: m.value for m in eval_b.metrics_rows}
        for name in sorted(set(metrics_a) | set(metrics_b)):
            metrics[name] = _delta(metrics_a.get(name), metrics_b.get(name))

    def latencies(rows: dict[str, dict[str, Any]]) -> list[float]:
        return [float(r["latency_ms"]) for r in rows.values() if r.get("latency_ms") is not None]

    return {
        "kind": "evaluation" if eval_a is not None and eval_b is not None else "prediction",
        "a": _side(pred_a, eval_a),
        "b": _side(pred_b, eval_b),
        "differs": differs,
        "metrics": metrics,
        "rows": {
            "aligned": len(aligned),
            "only_in_a": len(rows_a) - len(aligned),
            "only_in_b": len(rows_b) - len(aligned),
            "both_succeeded": len(both_ok),
            "agreements": len(agreements),
            "a_errored": sum(1 for r in rows_a.values() if r.get("error")),
            "b_errored": sum(1 for r in rows_b.values() if r.get("error")),
            "mean_latency_ms": _delta(_mean(latencies(rows_a)), _mean(latencies(rows_b))),
        },
        "changed_examples": [
            {
                "example_id": eid,
                "output_a": rows_a[eid].get("output"),
                "output_b": rows_b[eid].get("output"),
            }
            for eid in changed[:max_examples]
        ],
    }
