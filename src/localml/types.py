"""Shared platform primitives returned by the SDK.

These mirror the control-plane resources. They are intentionally thin data holders; the
server remains the source of truth. ``EvaluationJob`` and ``Deployment`` carry helper
methods that round-trip back through the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._polling import wait_for_terminal
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
class PromptVersion:
    """Versioned prompt template registered with the control plane."""

    id: str
    project: str
    name: str
    version: str
    template: str
    variables: list[str] = field(default_factory=list)
    _client: Client | None = field(default=None, repr=False, compare=False)

    def render(self, **variables: Any) -> str:
        """Render server-side with exactly the declared variables (missing/extra → error)."""
        if self._client is None:
            raise RuntimeError("prompt version is not bound to a client")
        return self._client.render_prompt(self.name, self.version, variables)


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

        Uses the shared exponential-backoff poller (capped at 10s). Raises
        :class:`EvaluationFailedError` if the job ends in ``failed`` or the timeout elapses.
        """
        try:
            status = wait_for_terminal(
                self.refresh,
                lambda: self.status,
                self._TERMINAL,
                timeout=timeout,
                poll_interval=poll_interval,
            )
        except TimeoutError as exc:
            raise EvaluationFailedError(
                f"evaluation {self.id} timed out (status={self.status})"
            ) from exc
        if status == "failed":
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
