"""Test fixtures.

Adds the API service package to ``sys.path`` so tests can import ``app`` directly, and
provides a FastAPI ``TestClient`` backed by a fresh in-memory SQLite database per test. The
``get_db`` dependency is overridden with a session bound to that database, so every test runs
against the real SQLAlchemy repository layer with full isolation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import StaticPool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "api"))


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    from app.db import Base
    from app.main import app
    from app.session import get_db
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[object]:
        db = TestingSession()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()
