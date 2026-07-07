"""Shared platform primitives returned by the SDK.

These mirror the control-plane resources. They are intentionally thin data holders; the
server remains the source of truth. ``EvaluationJob`` and ``Deployment`` carry helper
methods that round-trip back through the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._polling import wait_for_terminal
from .exceptions import EvaluationFailedError, PredictionFailedError

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
class PredictionJob:
    """Background job that runs a prompt over a dataset through an inference provider."""

    id: str
    model_version_id: str
    dataset_id: str
    prompt_version_id: str
    status: str = "queued"
    provider: str = "openai"
    completed_count: int = 0
    total_count: int = 0
    results_uri: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    _client: Client | None = field(default=None, repr=False, compare=False)

    _TERMINAL = frozenset({"completed", "failed"})

    def refresh(self) -> PredictionJob:
        """Fetch the latest job state from the control plane."""
        if self._client is None:
            return self
        latest = self._client.get_prediction(self.id)
        for name in (
            "status",
            "completed_count",
            "total_count",
            "results_uri",
            "summary",
            "error",
        ):
            setattr(self, name, getattr(latest, name))
        return self

    def wait(self, *, timeout: float = 600.0, poll_interval: float = 1.0) -> PredictionJob:
        """Poll until the job reaches a terminal state.

        Raises :class:`PredictionFailedError` if the job ends in ``failed`` or the timeout
        elapses.
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
            raise PredictionFailedError(
                f"prediction {self.id} timed out (status={self.status})"
            ) from exc
        if status == "failed":
            raise PredictionFailedError(f"prediction {self.id} failed: {self.error}")
        return self

    def results(self) -> list[dict[str, Any]]:
        """Fetch the per-example result records (input, rendered prompt, output, error)."""
        if self._client is None:
            raise RuntimeError("prediction job is not bound to a client")
        return self._client.get_prediction_results(self.id)


@dataclass
class EvaluationJob:
    """Background job that scores a completed prediction job's stored results."""

    id: str
    model_version_id: str | None = None
    prediction_job_id: str | None = None
    status: str = "queued"
    metrics: dict[str, float] | None = None
    report_uri: str | None = None
    error: str | None = None
    _client: Client | None = field(default=None, repr=False, compare=False)

    _TERMINAL = frozenset({"completed", "failed"})

    def refresh(self) -> EvaluationJob:
        """Fetch the latest job state from the control plane."""
        if self._client is None:
            return self
        latest = self._client.get_evaluation(self.id)
        for name in ("status", "metrics", "report_uri", "error"):
            setattr(self, name, getattr(latest, name))
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
            raise EvaluationFailedError(f"evaluation {self.id} failed: {self.error}")
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
