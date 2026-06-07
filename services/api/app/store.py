"""Process-local metadata store used by the control plane.

The SQLAlchemy models in :mod:`app.db` define the durable Postgres schema. This store keeps
tests and local development lightweight while preserving the same resource boundaries used
by the repository layer.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class Project:
    name: str
    description: str | None = None
    id: str = field(default_factory=_uuid)


@dataclass
class Run:
    project: str
    config: dict[str, Any]
    status: str = "running"
    mlflow_run_id: str | None = None
    id: str = field(default_factory=_uuid)
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelVersion:
    model_name: str
    version: int
    framework: str
    artifact_uri: str
    status: str = "created"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_uuid)


@dataclass
class Dataset:
    project: str
    name: str
    version: str
    artifact_uri: str
    row_count: int
    example_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_uuid)


@dataclass
class EvaluationJob:
    model_version_id: str
    dataset_uri: str
    requested_metrics: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    metrics: dict[str, float] | None = None
    report_uri: str | None = None
    id: str = field(default_factory=_uuid)


@dataclass
class Deployment:
    model_version_id: str
    target: str = "local"
    status: str = "active"
    endpoint_url: str | None = None
    id: str = field(default_factory=_uuid)


class Store:
    """Threadsafe in-memory collections keyed by id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.projects: dict[str, Project] = {}
        self.runs: dict[str, Run] = {}
        # model name -> list of versions (ordered)
        self.models: dict[str, list[ModelVersion]] = {}
        self.model_versions: dict[str, ModelVersion] = {}
        self.datasets: dict[str, list[Dataset]] = {}
        self.dataset_versions: dict[str, Dataset] = {}
        self.evaluations: dict[str, EvaluationJob] = {}
        self.deployments: dict[str, Deployment] = {}
        self.idempotency: dict[tuple[str, str], dict[str, Any]] = {}

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def next_version(self, model_name: str) -> int:
        return len(self.models.get(model_name, [])) + 1

    def next_dataset_version(self, name: str) -> str:
        return f"v{len(self.datasets.get(name, [])) + 1}"


store = Store()


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
