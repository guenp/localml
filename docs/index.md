# localml

A local ML experimentation platform demo that runs entirely on an Apple Silicon
workstation. It demonstrates the core architecture of a production ML platform at local
scale: a Python SDK, framework adapters, experiment tracking, a model registry, artifact
storage, evaluation jobs, and local model serving.

## Install

```sh
pip install localml
```

## Quickstart

```python
import localml as ml

ml.configure(api_url="http://localhost:8000", token="local-dev-token")

with ml.start_run(project="local-demo", config={"model": "tiny-llm"}):
    ml.log_params({"batch_size": 4})
    ml.log_metrics({"baseline_accuracy": 0.82})
```

## Architecture

```mermaid
flowchart LR
    User[SDK / CLI / Notebook] --> API[FastAPI control plane]
    API --> MLflow[MLflow<br/>tracking + registry]
    API --> DB[(Postgres<br/>metadata)]
    API --> Store[(MinIO<br/>artifacts)]
    API --> Queue[Redis<br/>job queue]
    API --> Serving[Local inference<br/>Ollama / MLX]
    Queue --> Worker[Worker]
    Worker --> Store
    Worker --> DB
```

The control plane is the source of truth for platform metadata. MLflow holds experiment
tracking state and model registry records, MinIO stores artifacts, and Redis carries
transient evaluation job state for the worker.
