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

## Job & deployment handles

- `EvaluationJob.wait(*, timeout=600.0, poll_interval=1.0)` — polls with exponential backoff;
  raises `EvaluationFailedError` on failure or timeout.
- `EvaluationJob.refresh()`
- `Deployment.predict(payload)`

## Exceptions

All derive from `localml.LocalMLError`: `AuthenticationError`, `ValidationError`,
`ArtifactUploadError`, `ModelRegistrationError`, `EvaluationFailedError`, `DeploymentError`.
