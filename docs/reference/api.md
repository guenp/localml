# API

The SDK is a functional, module-level surface (`import localml as ml`); the HTTPX client is an
internal transport configured via `ml.configure`.

## Configuration

- `localml.configure(api_url=None, token=None, *, timeout=None, max_retries=None)` — precedence:
  explicit args → env vars (`LOCALML_API_URL`, `LOCALML_API_TOKEN`) → `~/.localml/config.toml` →
  defaults.
- `localml.Config`

## Runs

- `localml.start_run(project, config=None)` — context manager; marks the run `completed` on
  clean exit, `failed` on exception.
- `localml.log_params(params)`
- `localml.log_metrics(metrics, *, step=None)`
- `localml.log_artifact(path, *, artifact_type="file")` — computes a SHA-256 and uploads to the
  pre-signed target when MinIO is available.

## Models

- `localml.register_model(name, artifact_uri, *, framework="generic", metadata=None)`
- `localml.evaluate(model, dataset, metrics)` → `EvaluationJob`
- `localml.deploy(model, target="local")` → `Deployment`

Framework adapters serialize + package a model and register it as a version:

- `localml.torch.log_model(model, name, *, example_input=None, metadata=None, save_dir=None)`
- `localml.jax.log_checkpoint(name, *, params=None, state=None, config=None, checkpoint_format="orbax", ...)`
- `localml.mlx.log_model(name, model_dir, *, quantization=None, metadata=None)`
- `localml.huggingface.log_pretrained(name, model_dir, *, metadata=None)`

## Datasets

- `localml.datasets.register(*, project, name, artifact_uri, rows=None, version=None, metadata=None)`
  → `Dataset` — assigns stable per-row `example_id`s.
- `localml.datasets.get(name)` → `list[Dataset]`

## Prompts

Versioned `str.format` templates with a sandboxed field grammar: only bare-identifier
placeholders (`{question}`) are accepted — attribute/index access and positional fields are
rejected at registration. Variables are auto-extracted server-side; rendering requires exactly
the declared variables (missing or extra → `ValidationError`).

- `localml.prompts.register(*, name, template, project="local", version=None, metadata=None)`
  → `PromptVersion` — versions auto-increment (`v1`, `v2`, …) unless `version` is given.
- `localml.prompts.get(name)` → `list[PromptVersion]`
- `localml.prompts.render(name, version, **variables)` → `str`
- `PromptVersion.render(**variables)` → `str` — renders server-side via the bound client.

## Predictions

Batch inference over a dataset, run on the background worker (decoupled from evaluation —
outputs are stored as JSONL and scored separately).

- `localml.predict(*, model, dataset, prompt, provider="openai", config=None)` →
  `PredictionJob` — `model`/`dataset`/`prompt` accept SDK objects, ids, or `name:version`.
  `provider` is `"openai"` (any OpenAI-compatible backend; `config["base_url"]` overrides the
  server's serving URL) or `"echo"` (deterministic, for smoke tests). `config` also carries
  generation parameters and `batch_size` (in-flight concurrency). Prompt variables are
  validated against the dataset's columns at submission (`ValidationError`).

## Job & deployment handles

- `PredictionJob.wait(*, timeout=600.0, poll_interval=1.0)` — raises `PredictionFailedError`
  on failure or timeout.
- `PredictionJob.results()` → `list[dict]` — per-example records: `example_id`, `input`,
  `rendered_prompt`, `output`, `latency_ms`, token counts, `error`.
- `PredictionJob.refresh()`
- `EvaluationJob.wait(*, timeout=600.0, poll_interval=1.0)` — polls with exponential backoff;
  raises `EvaluationFailedError` on failure or timeout.
- `EvaluationJob.refresh()`
- `Deployment.predict(payload)`

## Exceptions

All derive from `localml.LocalMLError`: `AuthenticationError`, `ValidationError`,
`ArtifactUploadError`, `ModelRegistrationError`, `PredictionFailedError`,
`EvaluationFailedError`, `DeploymentError`.
