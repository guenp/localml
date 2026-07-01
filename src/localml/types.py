"""Shared platform primitives returned by the SDK.

These mirror the control-plane resources. They are intentionally thin data holders; the
server remains the source of truth. ``EvaluationJob`` and ``Deployment`` carry helper
methods that round-trip back through the API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .exceptions import EvaluationFailedError

if TYPE_CHECKING:
    from .client import Client


@dataclass
class Run:
    """One tracked experiment execution."""

    id: str
    project: str
    status: str = "running"


@dataclass
class ModelVersion:
    """Immutable record of a specific model artifact and its metadata."""

    id: str
    model_name: str
    version: int
    framework: str
    artifact_uri: str
    status: str = "created"


@dataclass
class Dataset:
    """Versioned JSONL dataset registered with the control plane."""

    id: str
    project: str
    name: str
    version: str
    artifact_uri: str
    row_count: int
    example_ids: list[str] = field(default_factory=list)


@dataclass
class EvaluationJob:
    """Background job that evaluates a model version against a dataset."""

    id: str
    model_version_id: str
    status: str = "queued"
    metrics: dict[str, float] | None = None
    _client: Client | None = field(default=None, repr=False, compare=False)

    _TERMINAL = frozenset({"completed", "failed"})

    def refresh(self) -> EvaluationJob:
        """Fetch the latest job state from the control plane."""
        if self._client is None:
            return self
        latest = self._client.get_evaluation(self.id)
        self.status = latest.status
        self.metrics = latest.metrics
        return self

    def wait(self, *, timeout: float = 600.0, poll_interval: float = 1.0) -> EvaluationJob:
        """Poll until the job reaches a terminal state.

        Uses exponential backoff capped at 10s. Raises :class:`EvaluationFailedError`
        if the job ends in ``failed``.
        """
        deadline = time.monotonic() + timeout
        interval = poll_interval
        while self.status not in self._TERMINAL:
            if time.monotonic() > deadline:
                raise EvaluationFailedError(
                    f"evaluation {self.id} timed out (status={self.status})"
                )
            time.sleep(interval)
            interval = min(interval * 2, 10.0)
            self.refresh()
        if self.status == "failed":
            raise EvaluationFailedError(f"evaluation {self.id} failed")
        return self


@dataclass
class Deployment:
    """Active or historical serving configuration for a model version."""

    id: str
    model_version_id: str
    target: str = "local"
    status: str = "active"
    endpoint_url: str | None = None
    _client: Client | None = field(default=None, repr=False, compare=False)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a prediction request to the deployed model's endpoint."""
        if self._client is None:
            raise RuntimeError("deployment is not bound to a client")
        return self._client.predict(self.id, payload)
