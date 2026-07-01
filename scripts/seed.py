"""Seed a default local user and project directly in the database.

Idempotent: safe to run repeatedly. Points at ``DATABASE_URL`` (see ``.env`` /
``docker-compose.yml``). For a running control plane over HTTP, ``POST /projects`` also works,
but the default user is a DB-level concern so we seed it here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

from app.repositories import DEFAULT_PROJECT, get_or_create_project
from app.session import SessionLocal, init_db


def main() -> None:
    if _is_sqlite():
        init_db()
    with SessionLocal() as db:
        project = get_or_create_project(db, DEFAULT_PROJECT, "Default local project")
        db.commit()
        print(f"seeded default user 'local' and project '{project.name}' ({project.id})")


def _is_sqlite() -> bool:
    from app.config import settings

    return settings.database_url.startswith("sqlite")


if __name__ == "__main__":
    main()
