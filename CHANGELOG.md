# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Durable SQLAlchemy repository layer backing the control plane: routers use a request-scoped
  session (`app.session.get_db`); the in-memory store is removed. Postgres in production,
  SQLite for unit tests.
- DB-backed idempotency keys for every create operation (projects, runs, model versions,
  datasets, evaluations, deployments); replaying a key returns the original response, and
  reusing a key with a different body returns 409.
- Pre-signed MinIO upload URLs returned from `POST /runs/{id}/artifacts` and `POST /datasets`.
- Defensive MLflow wiring for run creation and model registration (no-ops when unreachable).
- `scripts/seed.py` (idempotent default user + project) and `scripts/reset.py` (fast in-place
  DB truncate + re-seed, or `--volumes` for a full Compose teardown).
- `CLAUDE.md` contributor/agent guide.

### Changed

- Local serving is now specified as an **OpenAI-compatible proxy** (Ollama / MLX-LM /
  llama.cpp / vLLM) rather than a bespoke inference service — see design §4.5 and roadmap
  Phases 3–4.

### Fixed

- MLflow integration hooks now fail fast (bounded `MLFLOW_HTTP_REQUEST_MAX_RETRIES`/timeout)
  so an unreachable tracking server no longer blocks the request path for minutes. Unit tests
  stub the optional-service hooks so they never perform network I/O.

## [0.1.0] - 2026-06-06

### Added

- Initial local ML experimentation platform scaffold.
