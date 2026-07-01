# CLAUDE.md

Guidance for AI agents and contributors working in this repo. Keep this file current.

## What this is

`localml` — a local ML experimentation platform for Apple Silicon: a Python SDK + FastAPI
control plane that mirror a production ML platform (tracking, model registry, artifacts,
datasets, evaluation jobs, local serving) at single-workstation scale. Full spec in
`docs/design.md`; phased plan and status in `ROADMAP.md`. Running notes (gitignored) in
`IMPLEMENTATION_LOG.md`.

## Layout

- `src/localml/` — SDK. `client.py` (HTTPX + retries + idempotency), `ops.py` / `run.py`
  (functional API: `ml.start_run`, `ml.log_metrics`, `ml.evaluate`, `ml.deploy`),
  `datasets.py`, framework adapters (`torch`/`jax`/`mlx`/`huggingface`), `config.py`,
  `types.py`, `exceptions.py`, `cli.py` (Typer).
- `services/api/app/` — control plane. `main.py` (app + lifespan), `db.py` (SQLAlchemy ORM =
  Postgres schema, source of truth), `session.py` (engine + `get_db`), `repositories.py`
  (idempotency, get-or-create, `name:version` resolution), `routers/`, `schemas.py` (Pydantic),
  `lifecycle.py` (state machine), `auth.py`, `queue.py` (Redis), `integrations.py`
  (MLflow/MinIO, all defensive), `config.py`, `alembic/`.
- `services/worker/` — Redis-consuming evaluation worker (real metrics land in Phase 3).
- `scripts/seed.py` / `scripts/reset.py`, `tests/`, `docs/`.

## Architecture notes

- **Metadata source of truth = Postgres via SQLAlchemy.** Routers depend on a request-scoped
  `Session` (`get_db`), which commits on success / rolls back on error. Repository helpers hold
  cross-cutting logic; simple CRUD is inline in routers.
- **Tests use SQLite** (in-memory, `StaticPool`) via a `get_db` dependency override in
  `tests/conftest.py`. The ORM uses only portable types. Postgres runs Alembic migrations;
  SQLite/local dev uses `session.init_db()` (called from the app lifespan for `sqlite://`).
- **Idempotency**: pass `Idempotency-Key` header on creates; stored in `idempotency_keys`.
  Same key + same body → original response; same key + different body → 409.
- **Optional services degrade gracefully.** MLflow, MinIO (boto3), and Redis all no-op when
  unavailable so the core flow works standalone and in tests. Don't make them hard deps.
- **`name:version`** (e.g. `assistant:v1`) resolves server-side to canonical ids for models and
  datasets (`POST /resolve`, and inline in create paths).
- **Serving = OpenAI-compatible proxy** (Ollama / MLX-LM / llama.cpp / vLLM), not a bespoke
  inference server. Implemented in Phase 4; deployment records already carry the endpoint.
- **Lifecycle**: `created → candidate → staging → production → deprecated → archived`
  (+ `failed`); only `staging`/`production` are deployable. Invalid transition → 422; wrong
  state for an action → 409.

## Commands (always use `uv`)

- `make test` / `uv run pytest` — tests. Add `-k name` to filter.
- `make lint` (`ruff check`) · `make fmt` (`ruff format`) · `make typecheck` (`ty check src/`).
- `make up` / `make down` — Docker Compose stack. `make seed` / `make reset`.
- Run API standalone against SQLite: `DATABASE_URL=sqlite:///localml.db uv run uvicorn
  app.main:app` from `services/api/` (tables auto-created on boot).

## Conventions

- Python ≥3.11, `from __future__ import annotations`, full type hints, ruff line length 100.
- **Test before and after meaningful changes.** Keep SDK and API route coverage healthy
  (target >80%, see roadmap Phase 6). New endpoints/behaviors get tests in `tests/`.
- Keep it lean: prefer well-established libraries (FastAPI, SQLAlchemy, Pydantic, MLflow,
  boto3, Redis) over new bespoke machinery.
- Update `ROADMAP.md` (checkboxes), `CHANGELOG.md` (`[Unreleased]`), and `IMPLEMENTATION_LOG.md`
  as you go. Don't commit/push unless asked.
- **Never add a `Co-Authored-By` trailer (or any co-authorship) to commits or PRs.**
