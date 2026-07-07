"""SQLAlchemy ORM models — the Postgres schema (source of truth for platform metadata).

This mirrors Section 5 of the design doc and is the source of truth for platform metadata.
Routers reach it through a request-scoped session (:func:`app.session.get_db`) and the
repository helpers in :mod:`app.repositories`. Postgres runs the Alembic migrations; SQLite
(unit tests / local dev) uses :func:`app.session.init_db`.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(Text, unique=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="project")
    models: Mapped[list[Model]] = relationship(back_populates="project")


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="runs")


class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    value: Mapped[float] = mapped_column(Float)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Param(Base):
    __tablename__ = "params"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    value: Mapped[str] = mapped_column(Text)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    uri: Mapped[str] = mapped_column(Text)
    artifact_type: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_model_project_name"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped[Project] = relationship(back_populates="models")
    versions: Mapped[list[ModelVersion]] = relationship(back_populates="model")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_version_model"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    framework: Mapped[str] = mapped_column(Text)
    artifact_uri: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, index=True)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    model: Mapped[Model] = relationship(back_populates="versions")


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("project_id", "name", "version", name="uq_dataset_version"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(Text)
    artifact_uri: Mapped[str] = mapped_column(Text)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    example_ids: Mapped[list] = mapped_column(JSON, default=list)
    # Keys present in *every* row (intersection), captured at registration when rows are
    # provided. Lets prediction jobs pre-flight prompt variables against the dataset.
    columns: Mapped[list] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    project: Mapped[Project] = relationship()


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("project_id", "name", "version", name="uq_prompt_version"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(Text)
    template: Mapped[str] = mapped_column(Text)
    variables: Mapped[list] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped[Project] = relationship()


class PredictionJob(Base):
    """Batch inference over a dataset: model + prompt + provider, run on the worker.

    Decoupled from evaluation (roadmap Phase 3): outputs are written once as a JSONL
    artifact (``results_uri``) and scored separately, so evals can re-run without
    re-inferring. ``completed_examples`` tracks per-row progress for resumable retries.
    """

    __tablename__ = "prediction_jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    prompt_version_id: Mapped[str] = mapped_column(ForeignKey("prompt_versions.id"), index=True)
    status: Mapped[str] = mapped_column(Text, index=True)
    provider: Mapped[str] = mapped_column(Text, default="openai")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    completed_examples: Mapped[list] = mapped_column(JSON, default=list)
    results_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    model_version: Mapped[ModelVersion] = relationship()
    dataset: Mapped[Dataset] = relationship()
    prompt_version: Mapped[PromptVersion] = relationship()


class EvaluationJob(Base):
    """Scores a completed prediction job's stored results against registered metrics.

    Phase 3 M3: keyed on ``prediction_job_id`` (evaluations re-run without re-inferring).
    ``model_version_id``/``dataset_id`` are denormalized from the prediction for querying;
    they stay nullable so legacy pre-M3 rows (bare model + dataset URI, record-only) remain
    valid. ``config["metrics"]`` holds the requested metric names.
    """

    __tablename__ = "evaluation_jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    prediction_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("prediction_jobs.id"), nullable=True, index=True
    )
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    status: Mapped[str] = mapped_column(Text, index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    report_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    prediction_job: Mapped[PredictionJob | None] = relationship()
    metrics_rows: Mapped[list[EvaluationMetric]] = relationship(back_populates="job")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    evaluation_job_id: Mapped[str] = mapped_column(ForeignKey("evaluation_jobs.id"), index=True)
    name: Mapped[str] = mapped_column(Text)
    value: Mapped[float] = mapped_column(Float)

    job: Mapped[EvaluationJob] = relationship(back_populates="metrics_rows")


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    target: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, index=True)
    endpoint_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Per-deployment backend overrides (base_url / model / api_key), resolved at proxy time —
    # hot model swap is a PATCH that updates these (or the target); no process restart.
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    model_version: Mapped[ModelVersion] = relationship()


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    actor_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(Text)
    resource_type: Mapped[str] = mapped_column(Text)
    resource_id: Mapped[str | None] = mapped_column(String, nullable=True)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    resource: Mapped[str] = mapped_column(Text, index=True)
    key: Mapped[str] = mapped_column(Text, index=True)
    request_hash: Mapped[str] = mapped_column(Text)
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("resource", "key", name="uq_idempotency_resource_key"),)
