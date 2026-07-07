"""HTTPX client for the FastAPI control plane.

Thin wrapper that handles auth headers, retries on transient HTTP failures, and maps
non-2xx responses to typed SDK exceptions. Create-style calls accept an idempotency key so
they can be safely retried.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from .config import Config, get_config
from .exceptions import (
    ArtifactUploadError,
    AuthenticationError,
    DeploymentError,
    LocalMLError,
    ModelRegistrationError,
    ValidationError,
)
from .types import (
    Dataset,
    Deployment,
    EvaluationJob,
    ModelVersion,
    PredictionJob,
    PromptVersion,
    Run,
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class Client:
    """Synchronous control-plane client."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        self._http = httpx.Client(
            base_url=self.config.api_url,
            headers=headers,
            timeout=self.config.timeout,
        )

    # -- low-level ---------------------------------------------------------------

    def _request(self, method: str, path: str, *, idempotent: bool = False, **kwargs: Any) -> Any:
        attempts = self.config.max_retries if (method == "GET" or idempotent) else 1
        if idempotent:
            kwargs.setdefault("headers", {})
            kwargs["headers"].setdefault("Idempotency-Key", str(uuid.uuid4()))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:  # network-level, retry
                last_exc = exc
                time.sleep(min(2**attempt, 5))
                continue
            if resp.status_code in _RETRYABLE_STATUS and attempt < attempts - 1:
                time.sleep(min(2**attempt, 5))
                continue
            return self._handle(resp)
        raise LocalMLError(f"request failed after {attempts} attempts: {last_exc}")

    @staticmethod
    def _handle(resp: httpx.Response) -> Any:
        if resp.is_success:
            return resp.json() if resp.content else None
        detail = resp.text
        if resp.status_code == 401:
            raise AuthenticationError(detail)
        if resp.status_code in (400, 422):
            raise ValidationError(detail)
        raise LocalMLError(f"HTTP {resp.status_code}: {detail}")

    # -- runs --------------------------------------------------------------------

    def create_run(self, project: str, config: dict[str, Any]) -> Run:
        data = self._request(
            "POST", "/runs", idempotent=True, json={"project": project, "config": config}
        )
        return Run(id=data["id"], project=data["project"], status=data.get("status", "running"))

    def get_run(self, run_id: str) -> Run:
        data = self._request("GET", f"/runs/{run_id}")
        return Run(id=data["id"], project=data["project"], status=data.get("status", "running"))

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None = None) -> None:
        self._request("POST", f"/runs/{run_id}/metrics", json={"metrics": metrics, "step": step})

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        self._request("POST", f"/runs/{run_id}/params", json={"params": params})

    def log_artifact(
        self, run_id: str, uri: str, artifact_type: str, checksum: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runs/{run_id}/artifacts",
            json={"uri": uri, "artifact_type": artifact_type, "checksum": checksum},
        )

    def upload_file(self, url: str, path: str) -> None:
        """PUT a local file to a pre-signed URL, raising :class:`ArtifactUploadError`.

        Streams from disk (bundles can be multi-GB) and retries transient failures — a
        pre-signed PUT is idempotent. Deliberately does not reuse ``self._http``: its
        ``Authorization`` header would clash with the URL's query-string signature.
        """
        attempts = max(1, self.config.max_retries)
        last_error: Exception | str | None = None
        for attempt in range(attempts):
            try:
                with Path(path).open("rb") as fh:
                    resp = httpx.put(url, content=fh, timeout=self.config.timeout)
            except httpx.TransportError as exc:  # network-level, retry
                last_error = exc
            except Exception as exc:
                raise ArtifactUploadError(f"failed to upload {path}: {exc}") from exc
            else:
                if resp.is_success:
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text}"
                if resp.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < attempts - 1:
                time.sleep(min(2**attempt, 5))
        raise ArtifactUploadError(f"failed to upload {path}: {last_error}")

    def complete_run(self, run_id: str, status: str) -> None:
        self._request("POST", f"/runs/{run_id}/metrics", json={"metrics": {}, "status": status})

    # -- models ------------------------------------------------------------------

    def register_model_version(
        self,
        name: str,
        framework: str,
        artifact_uri: str,
        metadata: dict[str, Any],
    ) -> ModelVersion:
        try:
            data = self._request(
                "POST",
                f"/models/{name}/versions",
                idempotent=True,
                json={
                    "model_name": name,
                    "framework": framework,
                    "artifact_uri": artifact_uri,
                    "metadata": metadata,
                },
            )
        except ValidationError as exc:
            raise ModelRegistrationError(str(exc)) from exc
        return ModelVersion(
            id=data["id"],
            model_name=data["model_name"],
            version=data["version"],
            framework=data["framework"],
            artifact_uri=data["artifact_uri"],
            status=data.get("status", "created"),
        )

    # -- datasets ---------------------------------------------------------------

    def register_dataset(
        self,
        *,
        project: str,
        name: str,
        artifact_uri: str,
        rows: list[dict[str, Any]] | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        data = self._request(
            "POST",
            "/datasets",
            idempotent=True,
            json={
                "project": project,
                "name": name,
                "artifact_uri": artifact_uri,
                "version": version,
                "rows": rows,
                "metadata": metadata or {},
            },
        )
        return Dataset(
            id=data["id"],
            project=data["project"],
            name=data["name"],
            version=data["version"],
            artifact_uri=data["artifact_uri"],
            row_count=data["row_count"],
            example_ids=data.get("example_ids", []),
        )

    def get_datasets(self, name: str) -> list[Dataset]:
        data = self._request("GET", f"/datasets/{name}")
        return [
            Dataset(
                id=item["id"],
                project=item["project"],
                name=item["name"],
                version=item["version"],
                artifact_uri=item["artifact_uri"],
                row_count=item["row_count"],
                example_ids=item.get("example_ids", []),
            )
            for item in data
        ]

    # -- prompts ------------------------------------------------------------------

    def register_prompt(
        self,
        *,
        name: str,
        template: str,
        project: str = "local",
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        data = self._request(
            "POST",
            "/prompts",
            idempotent=True,
            json={
                "name": name,
                "template": template,
                "project": project,
                "version": version,
                "metadata": metadata or {},
            },
        )
        return self._prompt_from(data)

    def get_prompts(self, name: str) -> list[PromptVersion]:
        data = self._request("GET", f"/prompts/{name}")
        return [self._prompt_from(item) for item in data]

    def render_prompt(self, name: str, version: str, variables: dict[str, Any]) -> str:
        data = self._request(
            "POST", f"/prompts/{name}/versions/{version}/render", json={"variables": variables}
        )
        return data["rendered"]

    def _prompt_from(self, data: dict[str, Any]) -> PromptVersion:
        return PromptVersion(
            id=data["id"],
            project=data["project"],
            name=data["name"],
            version=data["version"],
            template=data["template"],
            variables=data.get("variables", []),
            _client=self,
        )

    # -- predictions ---------------------------------------------------------------

    def create_prediction(
        self,
        *,
        model: str,
        dataset: str,
        prompt: str,
        provider: str = "openai",
        config: dict[str, Any] | None = None,
    ) -> PredictionJob:
        data = self._request(
            "POST",
            "/predictions",
            idempotent=True,
            json={
                "model": model,
                "dataset": dataset,
                "prompt": prompt,
                "provider": provider,
                "config": config or {},
            },
        )
        return self._prediction_from(data)

    def get_prediction(self, job_id: str) -> PredictionJob:
        return self._prediction_from(self._request("GET", f"/predictions/{job_id}"))

    def get_prediction_results(self, job_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/predictions/{job_id}/results")
        return data["results"]

    def _prediction_from(self, data: dict[str, Any]) -> PredictionJob:
        return PredictionJob(
            id=data["id"],
            model_version_id=data["model_version_id"],
            dataset_id=data["dataset_id"],
            prompt_version_id=data["prompt_version_id"],
            status=data.get("status", "queued"),
            provider=data.get("provider", "openai"),
            completed_count=data.get("completed_count", 0),
            total_count=data.get("total_count", 0),
            results_uri=data.get("results_uri"),
            summary=data.get("summary", {}),
            error=data.get("error"),
            _client=self,
        )

    # -- evaluations -------------------------------------------------------------

    def create_evaluation(
        self,
        model_version_id: str | None = None,
        dataset_uri: str | None = None,
        metrics: list[str] | None = None,
        *,
        prediction: str | None = None,
        config: dict[str, Any] | None = None,
        client_metrics: dict[str, float] | None = None,
    ) -> EvaluationJob:
        """Create an evaluation job: pass ``prediction`` (job id) or the legacy pair."""
        data = self._request(
            "POST",
            "/evaluations",
            idempotent=True,
            json={
                "prediction": prediction,
                "metrics": metrics or [],
                "config": config or {},
                "client_metrics": client_metrics or {},
                "model_version_id": model_version_id,
                "dataset_uri": dataset_uri,
            },
        )
        return self._evaluation_from(data)

    def get_evaluation(self, job_id: str) -> EvaluationJob:
        return self._evaluation_from(self._request("GET", f"/evaluations/{job_id}"))

    def _evaluation_from(self, data: dict[str, Any]) -> EvaluationJob:
        return EvaluationJob(
            id=data["id"],
            model_version_id=data.get("model_version_id"),
            prediction_job_id=data.get("prediction_job_id"),
            status=data.get("status", "queued"),
            metrics=data.get("metrics"),
            report_uri=data.get("report_uri"),
            error=data.get("error"),
            _client=self,
        )

    # -- deployments -------------------------------------------------------------

    def create_deployment(self, model_version_id: str, target: str) -> Deployment:
        try:
            data = self._request(
                "POST",
                "/deployments",
                idempotent=True,
                json={"model_version_id": model_version_id, "target": target},
            )
        except ValidationError as exc:
            raise DeploymentError(str(exc)) from exc
        return Deployment(
            id=data["id"],
            model_version_id=data["model_version_id"],
            target=data["target"],
            status=data.get("status", "active"),
            endpoint_url=data.get("endpoint_url"),
            _client=self,
        )

    def predict(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/deployments/{deployment_id}/predict", json=payload)

    def close(self) -> None:
        self._http.close()


_default_client: Client | None = None


def get_client() -> Client:
    """Return a process-wide default client, creating it on first use."""
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def reset_client() -> None:
    """Drop the cached default client (e.g. after reconfiguring)."""
    global _default_client
    if _default_client is not None:
        _default_client.close()
    _default_client = None
