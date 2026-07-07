"""prediction jobs

Revision ID: 0003_prediction_jobs
Revises: 0002_prompt_registry
Create Date: 2026-07-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_prediction_jobs"
down_revision = "0002_prompt_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_version_id", sa.String(), nullable=False),
        sa.Column("dataset_id", sa.String(), nullable=False),
        sa.Column("prompt_version_id", sa.String(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("completed_examples", sa.JSON(), nullable=False),
        sa.Column("results_uri", sa.Text(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["prompt_version_id"], ["prompt_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_prediction_jobs_model_version_id"),
        "prediction_jobs",
        ["model_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_prediction_jobs_dataset_id"), "prediction_jobs", ["dataset_id"], unique=False
    )
    op.create_index(
        op.f("ix_prediction_jobs_prompt_version_id"),
        "prediction_jobs",
        ["prompt_version_id"],
        unique=False,
    )
    op.create_index(op.f("ix_prediction_jobs_status"), "prediction_jobs", ["status"], unique=False)
    op.add_column(
        "datasets",
        sa.Column("columns", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )


def downgrade() -> None:
    op.drop_column("datasets", "columns")
    op.drop_index(op.f("ix_prediction_jobs_status"), table_name="prediction_jobs")
    op.drop_index(op.f("ix_prediction_jobs_prompt_version_id"), table_name="prediction_jobs")
    op.drop_index(op.f("ix_prediction_jobs_dataset_id"), table_name="prediction_jobs")
    op.drop_index(op.f("ix_prediction_jobs_model_version_id"), table_name="prediction_jobs")
    op.drop_table("prediction_jobs")
