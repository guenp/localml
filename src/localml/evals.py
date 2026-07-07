"""Evaluation helpers: score stored prediction results with registered metrics.

Built-in metrics (``exact_match``, ``contains_expected``, ``regex_match``, ``format_validity``,
``json_validity``, ``latency_p50/p95/p99``, ``error_rate``, ``avg_input/output_tokens``) run on
the control plane's worker. Custom metrics registered with :func:`register_metric` run
**client-side** — the server cannot import user code — so :func:`run` computes them from the
prediction's stored results up front and persists the values with the job.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .client import get_client
from .types import EvaluationJob, PredictionJob

MetricFn = Callable[[list[dict[str, Any]], dict[str, Any]], "float | None"]

_LOCAL_METRICS: dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn) -> None:
    """Register a client-side metric: ``fn(records, config) -> float | None``.

    ``records`` are the prediction's result rows (input, rendered_prompt, output, latency_ms,
    token counts, error); return ``None`` when nothing is scorable to omit the metric.
    """
    if not callable(fn):
        raise TypeError("metric must be callable")
    _LOCAL_METRICS[name] = fn


def run(
    prediction: PredictionJob | str,
    metrics: list[str],
    *,
    config: dict[str, Any] | None = None,
) -> EvaluationJob:
    """Queue an evaluation of a **completed** prediction job's stored results.

    Metric names registered with :func:`register_metric` are computed here from the stored
    results and persisted with the job; all other names are validated and computed
    server-side. ``config`` carries metric parameters (``expected_field``, ``pattern``, ...).
    """
    prediction_id = prediction.id if isinstance(prediction, PredictionJob) else prediction
    config = config or {}
    server_metrics = [name for name in metrics if name not in _LOCAL_METRICS]
    local_metrics = [name for name in metrics if name in _LOCAL_METRICS]

    client = get_client()
    client_values: dict[str, float] = {}
    if local_metrics:
        records = client.get_prediction_results(prediction_id)
        for name in local_metrics:
            value = _LOCAL_METRICS[name](records, config)
            if value is not None:
                client_values[name] = float(value)

    return client.create_evaluation(
        prediction=prediction_id,
        metrics=server_metrics,
        config=config,
        client_metrics=client_values,
    )
