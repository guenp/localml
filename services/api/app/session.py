"""Database engine, session factory, and the FastAPI ``get_db`` dependency.

Postgres in production; SQLite (in-memory) for unit tests. The ORM models in :mod:`app.db`
use only portable column types, so the same schema runs on both. Tests override ``get_db``
with a session bound to a throwaway SQLite engine (see ``tests/conftest.py``).
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import StaticPool, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .db import Base


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # In-memory SQLite lives inside a single connection, so share one via StaticPool;
        # file-backed SQLite just needs cross-thread access for the TestClient/uvicorn workers.
        in_memory = url in ("sqlite://", "sqlite:///:memory:")
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if in_memory:
            kwargs["poolclass"] = StaticPool
        return create_engine(url, future=True, **kwargs)
    return create_engine(url, future=True, pool_pre_ping=True)


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(bind: Engine | None = None) -> None:
    """Create all tables. Used for local SQLite/dev; Postgres uses Alembic migrations."""
    Base.metadata.create_all(bind or engine)


def get_db() -> Iterator[Session]:
    """Yield a request-scoped session, committing on success and rolling back on error."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
