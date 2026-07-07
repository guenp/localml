"""evaluation jobs keyed on prediction jobs

Revision ID: 0004_evaluation_prediction_link
Revises: 0003_prediction_jobs
Create Date: 2026-07-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_evaluation_prediction_link"
down_revision = "0003_prediction_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("evaluation_jobs", sa.Column("prediction_job_id", sa.String(), nullable=True))
    op.add_column("evaluation_jobs", sa.Column("error", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_evaluation_jobs_prediction_job_id",
        "evaluation_jobs",
        "prediction_jobs",
        ["prediction_job_id"],
        ["id"],
    )
    op.create_index(
        op.f("ix_evaluation_jobs_prediction_job_id"),
        "evaluation_jobs",
        ["prediction_job_id"],
        unique=False,
    )
    # Legacy pre-M3 rows required a model version; the prediction-keyed shape derives it
    # from the prediction job, so the column relaxes (its index exists since 0001).
    op.alter_column("evaluation_jobs", "model_version_id", nullable=True)


def downgrade() -> None:
    op.alter_column("evaluation_jobs", "model_version_id", nullable=False)
    op.drop_index(op.f("ix_evaluation_jobs_prediction_job_id"), table_name="evaluation_jobs")
    op.drop_constraint(
        "fk_evaluation_jobs_prediction_job_id", "evaluation_jobs", type_="foreignkey"
    )
    op.drop_column("evaluation_jobs", "error")
    op.drop_column("evaluation_jobs", "prediction_job_id")
