# Roadmap

High-level implementation steps beyond the initial scaffold. Items are roughly ordered but
not strictly sequential. The MVP target is a coherent local workflow on Apple Silicon.

The **inference/eval loop** (prompt → predict over a dataset → store outputs → score →
compare → iterate) is a first-class workflow, folded into Phases 1, 3, and 4 below rather
than bolted on as a parallel subsystem. Three corrections from design review are baked in:

- **Datasets, `name:version` resolution, and a namespaced client are Phase 1
  prerequisites**, not later integrations.
- **Prediction and evaluation are separate jobs.** Inference runs once and outputs are
  stored as a JSONL artifact; evaluation scores those stored outputs and can re-run without
  re-inferring. The schema change is additive (new `prediction_jobs` table; the existing
  `evaluate(model, dataset)` becomes sugar that creates a prediction job then an eval job).
- **Prediction loops run on the background worker**, never inline in the API request.

## Phase 0 — Scaffold (current)

- [x] Repository layout, packaging, tooling (ruff/mypy/pytest).
- [x] SDK public API surface and typed exceptions.
- [x] Framework adapter stubs (torch / jax / mlx / huggingface).
- [x] FastAPI control plane skeleton with route stubs.
- [x] Pydantic request/response schemas.
- [x] Docker Compose for the full local stack.
- [x] Design doc checked in.

## Phase 1 — Control plane core (foundations) ✅

- [x] Postgres schema via SQLAlchemy models + Alembic migrations.
- [x] Durable SQLAlchemy repository layer for projects, runs, metrics, params, artifacts,
      models, versions, datasets, and idempotency records. Routers depend on a request-scoped
      `Session` (`app.session.get_db`); the in-memory store is gone. Unit tests run against
      SQLite; Postgres in prod.
- [x] Wire MLflow client into run + model-version creation. Both are optional/defensive hooks
      (`create_mlflow_run`, `register_mlflow_model`) that degrade to no-ops when MLflow is
      unreachable, so the metadata flow never blocks on it.
- [x] MinIO artifact upload via **pre-signed PUT URLs**. `POST /runs/{id}/artifacts` and
      `POST /datasets` return an `upload_url` for the client to PUT bytes directly to MinIO
      (degrades to `None` when boto3/MinIO is unavailable).
- [x] Lifecycle state machine with validated transitions and 409/422 errors.
- [x] Idempotency keys for all create operations (projects, runs, model versions, datasets,
      evaluations, deployments), persisted in the `idempotency_keys` table; body-mismatch on a
      reused key is a 409.
- [x] Seed script: default local user + project (`scripts/seed.py`, idempotent) + fast reset
      (`scripts/reset.py` — in-place DB truncate + re-seed, or `--volumes` for a full Compose
      teardown).
- [x] **Dataset registry** — `POST /datasets` / `GET /datasets/{name}` + SDK
      (`ml.datasets.register/get`), pre-signed JSONL upload to MinIO, **stable per-row
      `example_id`s** (caller-supplied or content-hash; required for later comparison).
- [x] **`name:version` resolution** (e.g. `local-assistant:v1`) shared by models and datasets;
      resolves to canonical ids server-side. Prompts join this in Phase 3.

## Phase 2 — SDK end-to-end ✅

- [x] Decide and document **one SDK idiom** and apply it consistently: a functional,
      module-level surface (`ml.start_run`, `ml.evaluate`, `ml.deploy`, `ml.datasets.*`,
      framework adapters); the HTTPX `Client` stays an internal transport (see design §4.1).
      `ml.predict`/`ml.compare`/`ml.prompts.*` are added in Phase 3 on the same idiom.
- [x] Real HTTPX calls against the control plane with retry/backoff — exercised by the
      live-server `sdk` integration tests.
- [x] Run context manager logs to the live API (create → log → complete/fail).
- [x] Artifact staging + checksum before the registry record is finalized (SHA-256 +
      pre-signed upload when MinIO is available).
- [x] Adapter serialization: `base.package_dir` bundles model dirs (tar.gz + per-file manifest
      + content digest, symlinks dereferenced) and captures framework versions; mlx/huggingface
      fully; torch saves `state_dict` when torch is importable. (jax Orbax serialization +
      shape/dtype capture deferred to Phase 3 — params/state are flagged as provided only.)
- [x] Job-handle polling with exponential backoff (`.wait()`) shared via
      `_polling.wait_for_terminal` (ready for Phase 3 job handles).
- [x] Config precedence: explicit args → env vars → `~/.localml/config.toml` → defaults.
- [x] Contract tests pinning SDK payloads/routes to the OpenAPI schema.

## Phase 3 — Prediction + Evaluation loop

The core inference/eval workflow. Prediction and evaluation are decoupled so outputs can be
scored (and re-scored) without re-running inference, and variants can be compared.

### M1 — Prompt registry ✅

- [x] `PromptVersion` model + `POST /prompts` / `GET /prompts/{name}` (+ per-version GET and
      `POST .../render`); versioned templates with auto-incremented `v1`-style versions,
      idempotency keys, and `name:version` support in `/resolve`.
- [x] Template rendering with a **sandboxed** engine (explicit `str.format` restricted to bare
      identifiers — attribute/index access and positional fields rejected at registration);
      auto-extract `variables`; clear 422s on missing/extra variables.
- [x] SDK (`ml.prompts.register/get/render`, `PromptVersion.render(**vars)`) + CLI
      (`localml prompts register/get/render`).

### M2 — Prediction jobs (run on the worker) ✅

- [x] `PredictionJob` model: resolves model + dataset + prompt (ids or `name:version`) +
      inference config + provider; pre-flights prompt variables against the dataset's
      registered `columns` (422 before any inference). Alembic `0003_prediction_jobs`.
- [x] **`InferenceProvider` interface** (`generate(prompt, config) -> InferenceResult`). The
      default provider is a thin **OpenAI-compatible client** (httpx against a configurable
      `base_url`) — Ollama, MLX-LM, llama.cpp, and vLLM all speak the OpenAI
      `/v1/chat/completions` API, so one provider covers every local backend. No bespoke
      inference protocol. A deterministic `echo` provider covers pipeline smoke tests/CI.
- [x] Worker renders prompts per dataset row and runs inference; **`batch_size` =
      concurrency** of in-flight requests, not true batching. The worker runs from the API
      image (`python -m app.worker`); without Redis, jobs fall back to an in-process
      background thread so the standalone flow completes.
- [x] `PredictionResult` JSONL writer (input, rendered_prompt, output, latency, token
      counts, error) with a summary on the job (counts + duration); buffered per-batch
      appends, uploaded to MinIO when available.
- [x] Resumability + idempotent retries: track `completed_examples` (checkpointed per
      batch); errored examples still emit a result row so evals can score around them.
- [x] SDK `ml.predict(...)` handle with `.wait()` and `.results()`; CLI
      `predictions run/status/results`.

### M3 — Evaluation jobs (score stored results) ✅

- [x] `EvaluationJob` keyed on `prediction_job_id` (**additive** `0004` migration: legacy
      `model_version_id`/`dataset_id` columns relaxed to nullable; `evaluate(model, dataset,
      metrics, prompt=...)` becomes predict-then-eval sugar, and the pre-M3 record-only shape
      still works).
- [x] Pluggable metric registry (`app.evaluation.register_metric` server-side,
      `ml.evals.register_metric` client-side); built-ins: `exact_match`, `contains_expected`,
      `regex_match`, `format_validity`, `json_validity`, `latency_p50/p95/p99`, `error_rate`,
      `avg_input/output_tokens`. Custom metrics run client-side (the server can't import user
      code) and are persisted with the job.
- [x] Evaluation report written to the results dir (uploaded to MinIO when available); metric
      rows persisted to Postgres; metrics logged to MLflow defensively.
- [x] Job status transitions (queued → running → completed/failed) with traceback capture and
      bounded retries (deterministic failures skip the retry loop).
- [x] SDK (`ml.evals.run`) + CLI (`localml evals run/status`).

### M4 — Comparison reports ✅

- [x] Compare two prediction/eval jobs across aligned `example_id`s (model / prompt / config /
      provider / dataset variants): variant diff, row alignment (agreement + error counts,
      mean-latency delta), per-metric a/b/delta values for eval comparisons, capped
      changed-example samples. An evaluation reference compares through its prediction.
- [x] SDK (`ml.compare(...)` → `Comparison`) + CLI (`localml compare <a> <b>`) output.

## Phase 4 — Local serving, providers + deployment ✅

Serving is an **OpenAI-compatible proxy**, not a bespoke inference service: the control plane
resolves a deployment to a backend `base_url` + model id and forwards `/v1/chat/completions`
(and `/v1/completions`) to a local OpenAI-compatible server (Ollama / MLX-LM / llama.cpp /
vLLM). This reuses the Phase 3 provider wire format and keeps localml out of the
token-decoding business.

- [x] Thin proxy router: `POST /deployments/{id}/v1/chat/completions` and `/v1/completions`
      (plus `/predict` prompt→messages sugar) forward to the deployment's backend and return
      its reply. (The upstream response is buffered — adequate at single-workstation scale;
      SSE passthrough streaming is a future enhancement.)
- [x] Backend registry: map a deployment target to `{base_url, model, api_key?}`
      (`app.serving.register_backend`); **custom provider registration**
      (`ml.providers.register(name, base_url=, model=, api_key=)`) names a backend whose
      connection details expand into the deployment config.
- [x] Deployment flow: validate lifecycle state → resolve the backend → health-check it
      (`GET /v1/models`) → mark `active`/`degraded` (the proxy re-resolves per request, so a
      backend that comes up later works without redeploying).
- [x] `Deployment.predict()` / `.chat()` round-trip through the proxy.
- [x] Hot model swap = `PATCH /deployments/{id}` repoints the deployment's model version /
      target / backend config; no process restart.

## Phase 5 — Interfaces & DX

- [ ] Flesh out the Typer CLI (projects, runs, models, datasets, prompts, predictions,
      evals, compare, deployments).
- [ ] Optional Streamlit dashboard (projects, runs, metrics, predictions, evals, comparison,
      deploy, inference panel).
- [ ] Notebook quickstart + a **predict/eval loop notebook**: resolve a model, register a
      JSONL dataset, register two prompt versions, run predictions for both, evaluate, and
      compare to pick the better prompt/config.
- [ ] Generated and checked-in OpenAPI schema.

## Phase 6 — Quality & ops

- [ ] Docker Compose integration test stack (api + postgres + minio + redis).
- [ ] End-to-end smoke test (run → register → predict → eval → compare → promote → deploy →
      infer).
- [ ] Structured logging with request IDs; audit events for mutations.
- [ ] CI: lint → typecheck → unit → API → integration → package → smoke.
- [ ] Coverage gates (SDK >80%, API routes >80%).

## Later / out of MVP scope

- [ ] Advanced metrics: LLM-as-judge, pairwise preference, reward-model scoring, semantic
      similarity, human-review labels.
- [ ] Feedback-driven / RL workflows built on stored prediction results.
- [ ] OIDC auth, mTLS service-to-service, fine-grained RBAC.
- [ ] Encryption at rest, signed artifact manifests, secret management.
- [ ] Redis metadata cache, dataset cache, model warm pools.
- [ ] Full dataset versioning system.
- [ ] Migrate from Compose to local Kubernetes (kind/k3d/k3s); Helm chart.
- [ ] Horizontal scaling for API + workers; queue-based autoscaling.
- [ ] Multi-model serving routing; GPU/accelerator scheduling.
- [ ] Cloud object storage / managed Postgres / hosted tracking server.
