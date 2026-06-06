# localml

A local ML experimentation platform demo that runs on an Apple Silicon workstation.

!!! note
    These docs are built with [Zensical](https://zensical.org) and deployed to GitHub
    Pages on every push to `main`.

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
