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

## Phase 1 — Control plane core (foundations)

- [x] Postgres schema via SQLAlchemy models + Alembic migrations.
- [ ] Durable SQLAlchemy repository layer for projects, runs, metrics, params, artifacts,
      models, versions, datasets, and idempotency records. Current branch has process-local
      repository helpers for API behavior and tests.
- [ ] Wire MLflow client into run + model-version creation. Current branch has an optional,
      defensive run-creation hook; model-version registration still needs real MLflow
      integration.
- [ ] MinIO artifact upload (direct or pre-signed URL flow). Current branch has optional
      pre-signing plumbing; the API still needs a client-facing upload contract.
- [x] Lifecycle state machine with validated transitions and 409/422 errors.
- [ ] Idempotency keys for all create operations. Current branch covers projects, runs,
      model versions, and datasets; evaluations and deployments remain.
- [ ] Seed script: default local user + project + reset for local development.
- [ ] **Dataset registry** — `POST /datasets` / `GET /datasets/{name}` + SDK
      (`ml.datasets.register/get`), JSONL upload to MinIO, **stable per-row `example_id`s**
      (required for later comparison). Current branch covers the API, SDK namespace, and
      stable `example_id`s; JSONL upload to MinIO remains.
- [ ] **`name:version` resolution** (e.g. `local-assistant:v1`) shared by models, datasets,
      and prompts; resolves to canonical ids server-side. Current branch covers models and
      datasets; prompts remain for Phase 3.

## Phase 2 — SDK end-to-end

- [ ] Decide and document **one SDK idiom** and apply it consistently. Keep the functional
      surface (`ml.start_run`, `ml.evaluate`, `ml.predict`, `ml.compare`, `ml.prompts.*`)
      rather than introducing a parallel `Client()` object alongside it.
- [ ] Real HTTPX calls against the control plane with retry/backoff.
- [ ] Run context manager logs to the live API.
- [ ] Artifact staging + checksum before registry record is finalized.
- [ ] Adapter serialization: real `state_dict` / Orbax / MLX / safetensors packaging.
- [ ] Job-handle polling with exponential backoff (`.wait()`) shared by prediction + eval.
- [ ] Config precedence: env vars → `~/.localml/config.toml` → defaults.
- [ ] Contract tests pinning SDK payloads to the OpenAPI schema.

## Phase 3 — Prediction + Evaluation loop

The core inference/eval workflow. Prediction and evaluation are decoupled so outputs can be
scored (and re-scored) without re-running inference, and variants can be compared.

### M1 — Prompt registry

- [ ] `PromptVersion` model + `POST /prompts` / `GET /prompts/{name}`; versioned templates.
- [ ] Template rendering with a **sandboxed** engine (Jinja2 sandbox or explicit
      `str.format`); auto-extract `variables`; clear errors on missing/extra variables.
- [ ] SDK (`ml.prompts.register/get`) + CLI (`localml prompts register`).

### M2 — Prediction jobs (run on the worker)

- [ ] `PredictionJob` model: resolves model + dataset + prompt + inference config + provider.
- [ ] **`InferenceProvider` interface** (`generate(prompt, config) -> InferenceResult`) plus
      an **Ollama provider** for worker-side generation.
- [ ] Worker renders prompts per dataset row and runs inference; **`batch_size` =
      concurrency** of in-flight requests, not true batching.
- [ ] `PredictionResult` JSONL writer (input, rendered_prompt, output, latency, token
      counts, error) with an indexed summary; per-shard/buffered writes.
- [ ] Resumability + idempotent retries: track `completed_examples`; errored examples still
      emit a result row so evals can score around them.
- [ ] SDK handle with `.wait()` and `.results()`; CLI `predictions run/status/results`.

### M3 — Evaluation jobs (score stored results)

- [ ] `EvaluationJob` keyed on `prediction_job_id` (**additive** migration: keep legacy
      `model_version_id`/`dataset_id` columns nullable; existing `evaluate(model, dataset)`
      becomes predict-then-eval sugar).
- [ ] Pluggable metric registry (`ml.evals.register_metric`); built-ins:
      `exact_match`, `contains_expected`, `regex_match`, `format_validity`, `json_validity`,
      `latency_p50/p95/p99`, `error_rate`, `avg_input/output_tokens`.
- [ ] Evaluation report written to MinIO; metrics logged to MLflow + Postgres.
- [ ] Job status transitions (queued → running → completed/failed) with traceback capture
      and bounded retries.
- [ ] SDK (`ml.evals.run`) + CLI (`localml evals run`).

### M4 — Comparison reports

- [ ] Compare two prediction/eval jobs across aligned `example_id`s (model / prompt /
      config / dataset-slice variants).
- [ ] SDK (`ml.compare(...)`) + CLI (`localml compare <a> <b>`) summary output.

## Phase 4 — Local serving, providers + deployment

- [ ] **MLX-LM provider** and **custom provider registration**
      (`ml.providers.register("custom", fn)`), reusing the Phase 3 `InferenceProvider`
      interface.
- [ ] Inference service: `/load`, `/predict`, `/v1/chat/completions`, `/health`, `/models`.
- [ ] Deployment flow: validate lifecycle state → resolve artifact → load → mark active.
- [ ] `Deployment.predict()` round-trips through the serving runtime.
- [ ] Hot model swap on the serving process.

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
