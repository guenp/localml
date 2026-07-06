"""Optional service integrations used by the control plane.

Every function here degrades gracefully: if the optional dependency or backing service is
unavailable (common when running the API standalone or in unit tests), it logs and returns
``None`` so the core metadata flow keeps working. Real end-to-end wiring is exercised by the
Compose integration stack (Phase 6).
"""

from __future__ import annotations

import contextlib
import logging
import os

from .config import settings

log = logging.getLogger(__name__)

# Bound MLflow's HTTP client so an unreachable tracking server fails fast instead of retrying
# with exponential backoff for minutes (which would block the request path — and CI). Operators
# can still override these via the environment. ``setdefault`` runs at import, before any call.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")


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


def register_mlflow_model(name: str) -> str | None:
    """Ensure a registered model exists in MLflow and return its name.

    Idempotent: a model that already exists is treated as success.
    """
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("MLflow import failed: %s", exc)
        return None

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        with contextlib.suppress(Exception):  # already exists / transient
            client.create_registered_model(name)
        return name
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("MLflow model registration failed: %s", exc)
        return None


def upload_object(path: str, object_key: str) -> str | None:
    """Upload a local file to MinIO and return its ``s3://`` URI, or ``None`` if unavailable."""
    try:
        import boto3
    except ImportError:
        return None

    try:
        client = boto3.client("s3", endpoint_url=settings.minio_endpoint)
        client.upload_file(path, settings.minio_bucket, object_key)
        return f"s3://{settings.minio_bucket}/{object_key}"
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("MinIO upload failed: %s", exc)
        return None


def download_object(object_key: str, dest_path: str) -> bool:
    """Download a MinIO object to a local path. Returns False when unavailable."""
    try:
        import boto3
    except ImportError:
        return False

    try:
        client = boto3.client("s3", endpoint_url=settings.minio_endpoint)
        client.download_file(settings.minio_bucket, object_key, dest_path)
        return True
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("MinIO download failed: %s", exc)
        return False


def create_presigned_put_url(object_key: str) -> str | None:
    """Return a MinIO (S3) pre-signed PUT URL for ``object_key`` when boto3 is available."""
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
