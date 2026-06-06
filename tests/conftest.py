"""Test fixtures.

Adds the API service package to ``sys.path`` so tests can import ``app`` directly, and
provides a FastAPI ``TestClient`` with a fresh in-memory store per test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "api"))


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    from app.main import app
    from app.store import store
    from fastapi.testclient import TestClient

    # reset state between tests
    store.projects.clear()
    store.runs.clear()
    store.models.clear()
    store.model_versions.clear()
    store.evaluations.clear()
    store.deployments.clear()

    with TestClient(app) as c:
        yield c
