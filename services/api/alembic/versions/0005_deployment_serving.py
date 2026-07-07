"""deployment backend config for the serving proxy

Revision ID: 0005_deployment_serving
Revises: 0004_evaluation_prediction_link
Create Date: 2026-07-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_deployment_serving"
down_revision = "0004_evaluation_prediction_link"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "deployments",
        sa.Column("config", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )


def downgrade() -> None:
    op.drop_column("deployments", "config")
