"""Test fixtures.

Adds the API service package to ``sys.path`` so tests can import ``app`` directly. Two ways to
drive the control plane:

- ``client`` — a Starlette ``TestClient`` for fast in-process API tests.
- ``sdk`` — the real ``localml`` SDK pointed at the app running under a background uvicorn
  server, so SDK↔API tests exercise the actual HTTPX stack (retries, payloads, error mapping).

Both back onto a fresh in-memory SQLite database per test via a ``get_db`` dependency override.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "api"))


@pytest.fixture(autouse=True)
def offline_integrations(monkeypatch):  # type: ignore[no-untyped-def]
    """Neutralize optional-service hooks so unit tests never perform network I/O.

    In production these degrade gracefully on their own, but against an unreachable MLflow the
    HTTP client can retry for minutes. Tests patch the router-bound names to offline stubs so
    the suite stays fast and hermetic; real MLflow/MinIO wiring is covered by the Compose
    integration stack (roadmap Phase 6).
    """
    from app.routers import datasets, models, runs

    monkeypatch.setattr(runs, "create_mlflow_run", lambda *a, **k: None)
    monkeypatch.setattr(runs, "create_presigned_put_url", lambda *a, **k: None)
    monkeypatch.setattr(models, "register_mlflow_model", lambda *a, **k: None)
    monkeypatch.setattr(datasets, "create_presigned_put_url", lambda *a, **k: None)


@contextmanager
def _app_with_sqlite() -> Iterator[object]:
    """Yield the FastAPI app wired to a throwaway in-memory SQLite database."""
    from app.db import Base
    from app.main import app
    from app.session import get_db

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[object]:
        db = testing_session()
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
        yield app
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def _live_server(app: object) -> Iterator[str]:
    """Serve ``app`` on an ephemeral port via a background uvicorn thread."""
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on", ws="none"
    )
    server = uvicorn.Server(config)
    # Signal handlers can only be installed on the main thread; skip them in the server thread.
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started:
            if time.time() > deadline:  # pragma: no cover - startup failure
                raise RuntimeError("uvicorn did not start in time")
            time.sleep(0.02)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    with _app_with_sqlite() as app, TestClient(app) as c:
        yield c


@pytest.fixture
def sdk():  # type: ignore[no-untyped-def]
    """The real localml SDK, configured against a live in-process control plane."""
    import localml as ml
    from localml.client import get_client, reset_client

    with _app_with_sqlite() as app, _live_server(app) as base_url:
        reset_client()
        ml.configure(api_url=base_url, token=None, timeout=10, max_retries=3)
        try:
            yield get_client()
        finally:
            reset_client()
