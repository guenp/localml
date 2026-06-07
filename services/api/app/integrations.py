"""Optional service integrations used by the control plane."""

from __future__ import annotations

import logging

from .config import settings

log = logging.getLogger(__name__)


def create_mlflow_run(project: str) -> str | None:
    """Create a backing MLflow run when the dependency and service are available."""
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - depends on optional dependency state
        log.warning("MLflow import failed: %s", exc)
        return None

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(project)
        active = mlflow.start_run()
        mlflow.end_run()
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("MLflow run creation failed: %s", exc)
        return None
    return active.info.run_id


def create_presigned_put_url(object_key: str) -> str | None:
    """Return a MinIO pre-signed PUT URL when boto3 is available."""
    try:
        import boto3
    except ImportError:
        return None

    try:
        client = boto3.client("s3", endpoint_url=settings.minio_endpoint)
        return client.generate_presigned_url(
            "put_object",
            Params={"Bucket": settings.minio_bucket, "Key": object_key},
            ExpiresIn=3600,
        )
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("MinIO presign failed: %s", exc)
        return None
