"""Control-plane settings sourced from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://localml:localml@localhost:5432/localml"
    )
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    minio_endpoint: str = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
    minio_bucket: str = os.environ.get("MINIO_BUCKET", "localml-artifacts")
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    serving_url: str = os.environ.get("SERVING_URL", "http://localhost:11434")
    api_token: str = os.environ.get("LOCALML_API_TOKEN", "local-dev-token")
    auth_bypass: bool = os.environ.get("LOCALML_AUTH_BYPASS", "true").lower() == "true"


settings = Settings()
