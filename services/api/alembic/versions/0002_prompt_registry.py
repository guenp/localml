"""prompt registry

Revision ID: 0002_prompt_registry
Revises: 0001_initial_control_plane
Create Date: 2026-07-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_prompt_registry"
down_revision = "0001_initial_control_plane"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("variables", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", "version", name="uq_prompt_version"),
    )
    op.create_index(
        op.f("ix_prompt_versions_project_id"), "prompt_versions", ["project_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_prompt_versions_project_id"), table_name="prompt_versions")
    op.drop_table("prompt_versions")
