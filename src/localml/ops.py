"""Top-level lifecycle operations exposed on the ``localml`` namespace.

These are thin convenience wrappers that resolve the active run/client and delegate to the
control plane. Framework-specific model logging lives in :mod:`localml.adapters`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._hashing import sha256_file
from ._state import get_current_run
from .client import get_client
from .types import Deployment, EvaluationJob, ModelVersion


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
    returned (e.g. local dev without MinIO) the ``file://`` URI is recorded as-is.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    run = get_current_run()
    client = get_client()
    record = client.log_artifact(
        run.id,
        uri=p.resolve().as_uri(),
        artifact_type=artifact_type,
        checksum=sha256_file(p),
    )
    upload_url = record.get("upload_url") if record else None
    if upload_url:
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


def evaluate(
    model: ModelVersion | str,
    dataset: str,
    metrics: list[str],
) -> EvaluationJob:
    """Queue an evaluation job for a model version against a dataset."""
    model_version_id = model.id if isinstance(model, ModelVersion) else model
    return get_client().create_evaluation(
        model_version_id=model_version_id,
        dataset_uri=dataset,
        metrics=metrics,
    )


def deploy(model: ModelVersion | str, target: str = "local") -> Deployment:
    """Deploy a model version to a serving target (currently ``local``)."""
    model_version_id = model.id if isinstance(model, ModelVersion) else model
    return get_client().create_deployment(model_version_id=model_version_id, target=target)
