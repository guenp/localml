# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 3 M2 — Prediction jobs.** Batch inference decoupled from evaluation: `POST
  /predictions` resolves a model + dataset + prompt triple (ids or `name:version`),
  pre-flights the prompt's variables against the dataset's registered columns, and runs on
  the background worker via Redis (falling back to an in-process background thread when
  Redis is unavailable). The `InferenceProvider` interface ships an OpenAI-compatible
  default (httpx against any local backend's `/v1/chat/completions`) plus a deterministic
  `echo` provider; `batch_size` bounds in-flight concurrency. Results append to a JSONL
  file (input, rendered prompt, output, latency, token counts, per-example error) uploaded
  to MinIO when available, with per-batch `completed_examples` checkpoints so re-runs skip
  finished examples. SDK `ml.predict(...)` → `PredictionJob.wait()/.results()`; CLI
  `localml predictions run/status/results`; datasets now record their common `columns`.
  Alembic migration `0003_prediction_jobs`.
- **Phase 3 M1 — Prompt registry.** Versioned `str.format` prompt templates with a sandboxed
  field grammar (bare identifiers only; attribute/index access rejected at registration) and
  server-side variable extraction. `POST /prompts` (idempotent, auto-versioned) /
  `GET /prompts/{name}` / `POST /prompts/{name}/versions/{version}/render` (422 on
  missing/extra variables), prompts join `name:version` resolution in `/resolve`, plus the
  `ml.prompts` SDK namespace (`register`/`get`/`render`, `PromptVersion.render(**vars)`) and a
  `localml prompts` CLI group. Alembic migration `0002_prompt_registry`.
- **Phase 2 — SDK end-to-end.** SDK↔API integration tests drive the real `localml` SDK against
  a live in-process control plane (background uvicorn), plus an OpenAPI contract test pinning
  the SDK's routes to the schema.
- Artifact staging computes a SHA-256 checksum and uploads bytes to a pre-signed MinIO target
  when available (`client.upload_file`, `ArtifactUploadError`). Uploads stream from disk and
  retry transient failures; when an upload target exists the artifact record stores the
  object-store URI (not the client-local path). `ml.log_artifact` also accepts directories,
  bundling them like the adapters do.
- Framework adapters package model directories into checksummed `.tar.gz` bundles with a
  per-file manifest and capture framework versions (`base.package_dir`, `base.framework_version`,
  shared `base.package_and_register`). Symlinked files are dereferenced into the bundle, user
  metadata cannot override the derived `checksum`/`manifest`, and a failed `torch.save` raises
  instead of silently registering a weightless bundle.
- Shared job-handle polling (`_polling.wait_for_terminal`) behind `EvaluationJob.wait`, ready
  for Phase 3 prediction jobs; `client.get_run`.
- Documented the single functional SDK idiom (design §4.1).

_Phase 1 (control plane core):_

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

- The background worker now runs from the API image (`python -m app.worker`, consuming both
  the prediction and evaluation queues); the separate `services/worker/` scaffold is removed.
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
