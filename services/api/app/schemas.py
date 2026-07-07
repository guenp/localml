"""Pydantic request/response schemas for the control plane API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# -- requests ------------------------------------------------------------------


class CreateProjectRequest(BaseModel):
    name: str
    description: str | None = None


class CreateRunRequest(BaseModel):
    project: str
    config: dict[str, Any] = Field(default_factory=dict)


class LogMetricsRequest(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    step: int | None = None
    status: str | None = None


class LogParamsRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class LogArtifactRequest(BaseModel):
    uri: str
    artifact_type: str = "file"
    checksum: str | None = None


class ArtifactResponse(BaseModel):
    id: str
    uri: str
    artifact_type: str
    checksum: str | None = None
    upload_url: str | None = None


class RegisterModelVersionRequest(BaseModel):
    model_name: str
    framework: str
    artifact_uri: str
    project: str = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromoteRequest(BaseModel):
    target_status: str


class CreateEvaluationRequest(BaseModel):
    # Phase 3 shape: score a completed prediction job's stored results.
    prediction: str | None = None  # prediction-job id
    metrics: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    # Custom metrics computed client-side by the SDK (the server can't run user code);
    # persisted with the job alongside the worker-computed built-ins.
    client_metrics: dict[str, float] = Field(default_factory=dict)
    # Legacy pre-M3 shape (record-only; kept for compatibility).
    model_version_id: str | None = None
    dataset_uri: str | None = None


class CreateDeploymentRequest(BaseModel):
    model_version_id: str
    target: str = "local"


class RegisterDatasetRequest(BaseModel):
    project: str
    name: str
    artifact_uri: str
    version: str | None = None
    rows: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegisterPromptRequest(BaseModel):
    name: str
    template: str
    project: str = "local"
    version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderPromptRequest(BaseModel):
    variables: dict[str, Any] = Field(default_factory=dict)


class CreatePredictionRequest(BaseModel):
    model: str  # model-version id or name:version
    dataset: str  # dataset id or name:version
    prompt: str  # prompt-version id or name:version
    provider: str = "openai"
    config: dict[str, Any] = Field(default_factory=dict)


class ResolveReferenceRequest(BaseModel):
    resource_type: str
    reference: str


# -- responses -----------------------------------------------------------------


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None = None


class RunResponse(BaseModel):
    id: str
    project: str
    status: str
    config: dict[str, Any] = Field(default_factory=dict)


class ModelVersionResponse(BaseModel):
    id: str
    model_name: str
    version: int
    framework: str
    artifact_uri: str
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    name: str
    versions: list[ModelVersionResponse] = Field(default_factory=list)


class EvaluationJobResponse(BaseModel):
    id: str
    prediction_job_id: str | None = None
    model_version_id: str | None = None
    status: str
    metrics: dict[str, float] | None = None
    report_uri: str | None = None
    error: str | None = None


class PredictionJobResponse(BaseModel):
    id: str
    model_version_id: str
    dataset_id: str
    prompt_version_id: str
    status: str
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)
    completed_count: int = 0
    total_count: int = 0
    results_uri: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PredictionResultsResponse(BaseModel):
    job_id: str
    results: list[dict[str, Any]] = Field(default_factory=list)


class DeploymentResponse(BaseModel):
    id: str
    model_version_id: str
    target: str
    status: str
    endpoint_url: str | None = None


class DatasetResponse(BaseModel):
    id: str
    project: str
    name: str
    version: str
    artifact_uri: str
    row_count: int
    example_ids: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    upload_url: str | None = None


class PromptVersionResponse(BaseModel):
    id: str
    project: str
    name: str
    version: str
    template: str
    variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderPromptResponse(BaseModel):
    name: str
    version: str
    rendered: str


class ComparisonSide(BaseModel):
    job_id: str  # the referenced job (evaluation id when the reference was an evaluation)
    prediction_job_id: str
    model_version_id: str
    prompt_version_id: str
    dataset_id: str
    provider: str
    metrics: dict[str, float] | None = None  # present when the side is an evaluation


class ValueDelta(BaseModel):
    a: float | None = None
    b: float | None = None
    delta: float | None = None  # b - a when both sides have a value


class ComparisonRows(BaseModel):
    aligned: int
    only_in_a: int
    only_in_b: int
    both_succeeded: int
    agreements: int  # aligned rows where both succeeded with identical outputs
    a_errored: int
    b_errored: int
    mean_latency_ms: ValueDelta


class ChangedExample(BaseModel):
    example_id: str
    output_a: str | None = None
    output_b: str | None = None


class ComparisonResponse(BaseModel):
    kind: str  # "evaluation" when both references are evaluation jobs, else "prediction"
    a: ComparisonSide
    b: ComparisonSide
    differs: list[str] = Field(default_factory=list)
    metrics: dict[str, ValueDelta] = Field(default_factory=dict)
    rows: ComparisonRows
    changed_examples: list[ChangedExample] = Field(default_factory=list)


class ResolveReferenceResponse(BaseModel):
    resource_type: str
    reference: str
    id: str
    name: str
    version: str
