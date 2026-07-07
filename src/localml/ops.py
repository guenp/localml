"""Top-level lifecycle operations exposed on the ``localml`` namespace.

These are thin convenience wrappers that resolve the active run/client and delegate to the
control plane. Framework-specific model logging lives in :mod:`localml.adapters`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import evals
from ._hashing import sha256_file
from ._state import get_current_run
from .adapters import base
from .client import get_client
from .types import (
    Comparison,
    Dataset,
    Deployment,
    EvaluationJob,
    ModelVersion,
    PredictionJob,
    PromptVersion,
)


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    """Log scalar metrics to the active run."""
    run = get_current_run()
    get_client().log_metrics(run.id, metrics, step=step)


def log_params(params: dict[str, Any]) -> None:
    """Log hyperparameters / config values to the active run."""
    run = get_current_run()
    get_client().log_params(run.id, params)


def log_artifact(path: str, *, artifact_type: str = "file") -> None:
    """Stage a local artifact for the active run.

    Computes a SHA-256 checksum, finalizes the registry record (which returns a pre-signed
    upload target when MinIO is available), and uploads the bytes. When no upload target is
    returned (e.g. local dev without MinIO) the ``file://`` URI is recorded as-is. A directory
    is bundled into a ``.tar.gz`` first (same packaging as the framework adapters), with the
    manifest-derived content digest as its checksum.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    if p.is_dir():
        p, checksum, _ = base.package_dir(p)
    else:
        checksum = sha256_file(p)
    run = get_current_run()
    client = get_client()
    record = client.log_artifact(
        run.id,
        uri=base.stage_artifact(p),
        artifact_type=artifact_type,
        checksum=checksum,
    )
    if upload_url := record.get("upload_url"):
        client.upload_file(upload_url, str(p))


def register_model(
    name: str,
    artifact_uri: str,
    *,
    framework: str = "generic",
    metadata: dict[str, Any] | None = None,
) -> ModelVersion:
    """Register a new model version from an already-staged artifact URI."""
    return get_client().register_model_version(
        name=name,
        framework=framework,
        artifact_uri=artifact_uri,
        metadata=metadata or {},
    )


def predict(
    *,
    model: ModelVersion | str,
    dataset: Dataset | str,
    prompt: PromptVersion | str,
    provider: str = "openai",
    config: dict[str, Any] | None = None,
) -> PredictionJob:
    """Queue a prediction job: render ``prompt`` per dataset row and run inference.

    ``model``/``dataset``/``prompt`` accept SDK objects, canonical ids, or ``name:version``
    references. The default provider talks to any OpenAI-compatible backend (Ollama, MLX-LM,
    llama.cpp, vLLM) at ``config["base_url"]`` (falling back to the server's configured
    serving URL); ``config`` also carries generation parameters (``temperature``,
    ``max_tokens``, ...) and ``batch_size`` — the number of concurrent in-flight requests.
    """
    return get_client().create_prediction(
        model=model.id if isinstance(model, ModelVersion) else model,
        dataset=dataset.id if isinstance(dataset, Dataset) else dataset,
        prompt=prompt.id if isinstance(prompt, PromptVersion) else prompt,
        provider=provider,
        config=config,
    )


def evaluate(
    model: ModelVersion | str,
    dataset: Dataset | str,
    metrics: list[str],
    *,
    prompt: PromptVersion | str | None = None,
    provider: str = "openai",
    config: dict[str, Any] | None = None,
    predict_timeout: float = 600.0,
) -> EvaluationJob:
    """Evaluate a model on a dataset — predict-then-eval sugar.

    With ``prompt`` this queues a prediction job (``ml.predict``), waits for it to complete
    (up to ``predict_timeout`` seconds), then queues an evaluation over the stored results —
    equivalent to ``ml.evals.run(ml.predict(...).wait(), metrics)``. ``config`` is shared:
    generation parameters and ``batch_size`` go to the prediction; metric parameters
    (``expected_field``, ``pattern``, ...) to the evaluation.

    Without ``prompt`` it falls back to the legacy record-only evaluation shape (kept for
    compatibility; those jobs have no stored results to score).
    """
    if prompt is not None:
        job = predict(model=model, dataset=dataset, prompt=prompt, provider=provider, config=config)
        job.wait(timeout=predict_timeout)
        return evals.run(job, metrics, config=config)
    model_version_id = model.id if isinstance(model, ModelVersion) else model
    dataset_uri = dataset.id if isinstance(dataset, Dataset) else dataset
    return get_client().create_evaluation(
        model_version_id=model_version_id,
        dataset_uri=dataset_uri,
        metrics=metrics,
    )


def compare(
    a: PredictionJob | EvaluationJob | str,
    b: PredictionJob | EvaluationJob | str,
    *,
    max_examples: int = 20,
) -> Comparison:
    """Compare two prediction/evaluation jobs across aligned ``example_id``s.

    Pass job handles or ids. When both are evaluation jobs the report includes per-metric
    a/b/delta values; row alignment always comes from the underlying predictions' stored
    results. ``max_examples`` caps the changed-example samples returned.
    """
    ref_a = a if isinstance(a, str) else a.id
    ref_b = b if isinstance(b, str) else b.id
    return get_client().compare(ref_a, ref_b, max_examples=max_examples)


def deploy(
    model: ModelVersion | str,
    target: str = "local",
    *,
    provider: str | None = None,
    config: dict[str, Any] | None = None,
) -> Deployment:
    """Deploy a model version to a serving target.

    ``provider`` names a backend registered with :func:`ml.providers.register`; its
    connection details (``base_url``/``model``/``api_key``) are merged into ``config``.
    ``config`` carries backend overrides resolved at proxy time; when neither is given the
    target's registered backend (or the server's serving URL) is used. The returned
    deployment is ``active`` when the backend answered a health check, ``degraded`` otherwise.
    """
    from . import providers

    model_version_id = model.id if isinstance(model, ModelVersion) else model
    merged = dict(config or {})
    if provider is not None:
        merged = {**providers.config_for(provider), **merged}
    return get_client().create_deployment(
        model_version_id=model_version_id, target=target, config=merged
    )
