# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 5 — Interfaces & DX.** The Typer CLI now covers every resource group's endpoints:
  `projects create/get`, `datasets register/get/version`, `models version/promote`, and
  `deployments get/delete` join the existing prompts/predictions/evals/compare/deployments
  commands, plus top-level `localml version` and `localml dashboard`. An optional Streamlit
  dashboard (`localml dashboard`, gated behind the `dashboard` extra) inspects runs, prediction
  jobs, evaluations and comparisons and routes a prompt through a deployment's serving proxy,
  all over the same HTTP client the SDK/CLI use. Two quickstart notebooks ship in `notebooks/`
  — a full SDK tour (`quickstart.ipynb`) and the predict → evaluate → compare loop
  (`predict_eval_loop.ipynb`). The control plane's OpenAPI schema is generated and checked in
  (`docs/openapi.json` via `scripts/export_openapi.py`), with a test that fails if it drifts
  from the live app.
- **Phase 4 — Local serving proxy.** Serving is an OpenAI-compatible proxy, not a bespoke
  inference server. A backend registry maps a deployment target to a `{base_url, model,
  api_key}`, resolved at request time (deployment `config` → registered backend → the global
  serving URL, model id defaulting to the deployed model's registry name). Creating a
  deployment validates lifecycle state, health-checks the backend (`GET /v1/models`), and
  records `active`/`degraded`. `POST /deployments/{id}/v1/chat/completions` and
  `/v1/completions` forward to the backend and return its reply; `/predict` is prompt→messages
  sugar. `PATCH /deployments/{id}` hot-swaps the model version, target, or backend config with
  no restart. SDK: `Deployment.chat()/.predict()/.swap()`, `ml.deploy(..., provider=, config=)`,
  the `ml.providers` registry (`register`/`get`/`config_for`); CLI `localml deployments
  create/swap/predict`. Additive Alembic migration `0005_deployment_serving`.
- **Phase 3 M4 — Comparison reports.** `GET /compare?a=&b=` compares two prediction- or
  evaluation-job references across aligned `example_id`s: what varied between the variants
  (model / prompt / dataset / provider / config), row alignment (aligned/only-in counts,
  output agreement, error counts, mean-latency delta), per-metric a/b/delta values when both
  are evaluations, and capped changed-example samples. SDK `ml.compare(a, b)` → `Comparison`;
  CLI `localml compare`.
- **Phase 3 M3 — Evaluation jobs.** Evaluations are keyed on a completed prediction job and
  score its stored JSONL results (re-runnable without re-inferring). A pluggable metric
  registry ships `exact_match`, `contains_expected`, `regex_match`, `format_validity`,
  `json_validity`, `latency_p50/p95/p99`, `error_rate`, and `avg_input/output_tokens`; unknown
  metrics or a missing `regex_match` pattern are rejected (422) at create. The worker scores on
  a background thread (or Redis), captures tracebacks with bounded retries for transient
  failures, writes a JSON report (uploaded to MinIO when available), persists metric rows, and
  logs to MLflow defensively. SDK `ml.evals.run(...)` + `ml.evals.register_metric` (custom
  metrics computed client-side), `ml.evaluate(..., prompt=...)` as predict-then-eval sugar;
  CLI `localml evals run/status`. Additive Alembic migration `0004_evaluation_prediction_link`
  (evaluations gain a nullable `prediction_job_id`; the legacy `model_version_id` +
  `dataset_uri` record-only shape still works).
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
- Local serving is now an **OpenAI-compatible proxy** (Ollama / MLX-LM / llama.cpp / vLLM)
  rather than a bespoke inference service — implemented in Phase 4 (see design §4.5).
- The in-process no-Redis job fallback is shared by prediction and evaluation jobs
  (`app.background.schedule_inline`).

### Fixed

- MLflow integration hooks now fail fast (bounded `MLFLOW_HTTP_REQUEST_MAX_RETRIES`/timeout)
  so an unreachable tracking server no longer blocks the request path for minutes. Unit tests
  stub the optional-service hooks so they never perform network I/O.

## [0.1.0] - 2026-06-06

### Added

- Initial local ML experimentation platform scaffold.
