# API

## Configuration

- `localml.configure(api_url, token=None, timeout=30.0)`
- `localml.Config`

## Runs

- `localml.start_run(project, config=None)`
- `localml.log_params(params)`
- `localml.log_metrics(metrics, step=None)`
- `localml.log_artifact(path, artifact_type="file")`

## Models

- `localml.register_model(name, framework, artifact_uri, metadata=None)`
- `localml.evaluate(model, dataset, metrics)`
- `localml.deploy(model, target="local")`

Framework adapters are exposed as `localml.torch`, `localml.jax`, `localml.mlx`, and
`localml.huggingface`.
