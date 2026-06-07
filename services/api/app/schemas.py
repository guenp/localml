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


class RegisterModelVersionRequest(BaseModel):
    model_name: str
    framework: str
    artifact_uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromoteRequest(BaseModel):
    target_status: str


class CreateEvaluationRequest(BaseModel):
    model_version_id: str
    dataset_uri: str
    metrics: list[str]
    config: dict[str, Any] = Field(default_factory=dict)


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
    model_version_id: str
    status: str
    metrics: dict[str, float] | None = None
    report_uri: str | None = None


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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolveReferenceResponse(BaseModel):
    resource_type: str
    reference: str
    id: str
    name: str
    version: str
