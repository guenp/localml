"""Test fixtures.

Adds the API service package to ``sys.path`` so tests can import ``app`` directly. Two ways to
drive the control plane:

- ``client`` — a Starlette ``TestClient`` for fast in-process API tests.
- ``sdk`` — the real ``localml`` SDK pointed at the app running under a background uvicorn
  server, so SDK↔API tests exercise the actual HTTPX stack (retries, payloads, error mapping).

Both back onto a fresh in-memory SQLite database per test via a ``get_db`` dependency override.
"""

from __future__ import annotations

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
def _app_with_sqlite(url: str = "sqlite://") -> Iterator[object]:
    """Yield the FastAPI app wired to a throwaway SQLite database.

    In-memory (the default) uses a single shared connection (StaticPool) — fine for the
    single-threaded ``TestClient``. The live server runs endpoints across a threadpool, so it
    passes a file URL instead, giving each request its own connection with normal commit
    visibility (mirroring Postgres) rather than racing on one shared in-memory connection.
    """
    from app.db import Base
    from app.main import app
    from app.session import get_db

    engine_kwargs: dict = {"connect_args": {"check_same_thread": False}, "future": True}
    if url in ("sqlite://", "sqlite:///:memory:"):
        engine_kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **engine_kwargs)
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


@contextmanager
def _live_server(app: object) -> Iterator[str]:
    """Serve ``app`` on an ephemeral port via a background uvicorn thread."""
    import uvicorn

    # port=0 lets the kernel assign a free port at bind time — no find-then-rebind race.
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", lifespan="on", ws="none"
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
        port = server.servers[0].sockets[0].getsockname()[1]
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
def sdk(tmp_path):  # type: ignore[no-untyped-def]
    """The real localml SDK, configured against a live in-process control plane.

    Uses a file-backed SQLite DB so the threaded server gets a connection per request and
    read-your-writes holds across separate SDK calls.
    """
    import localml as ml
    from localml.client import get_client, reset_client

    url = f"sqlite:///{tmp_path / 'localml_test.db'}"
    with _app_with_sqlite(url) as app, _live_server(app) as base_url:
        reset_client()
        ml.configure(api_url=base_url, token=None, timeout=10, max_retries=3)
        try:
            yield get_client()
        finally:
            reset_client()
