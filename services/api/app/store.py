"""In-memory store for the scaffold.

Lets the API run end-to-end without Postgres so the demo flow works out of the box. Phase 1
replaces this with a SQLAlchemy repository layer backed by the ORM models in :mod:`app.db`.
All data is process-local and lost on restart.
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
        self.evaluations: dict[str, EvaluationJob] = {}
        self.deployments: dict[str, Deployment] = {}

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def next_version(self, model_name: str) -> int:
        return len(self.models.get(model_name, [])) + 1


store = Store()


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
