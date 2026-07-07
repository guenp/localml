"""localml — Local ML experimentation platform SDK.

Public API::

    import localml as ml

    ml.configure(api_url="http://localhost:8000", token="local-dev-token")

    with ml.start_run(project="local", config={"lr": 0.001}) as run:
        ml.log_metrics({"accuracy": 0.91})
        ml.log_artifact("outputs/model.safetensors")
        version = ml.mlx.log_model(name="assistant", model_dir="./model")
        job = ml.evaluate(model=version, dataset="datasets/eval.jsonl", metrics=["accuracy"])
        job.wait()
        ml.deploy(model=version, target="local")
"""

from __future__ import annotations

from . import datasets, evals, huggingface, jax, mlx, prompts, torch
from .config import Config, configure
from .exceptions import (
    ArtifactUploadError,
    AuthenticationError,
    DeploymentError,
    EvaluationFailedError,
    LocalMLError,
    ModelRegistrationError,
    PredictionFailedError,
    ValidationError,
)
from .ops import deploy, evaluate, log_artifact, log_metrics, log_params, predict, register_model
from .run import start_run
from .types import (
    Dataset,
    Deployment,
    EvaluationJob,
    ModelVersion,
    PredictionJob,
    PromptVersion,
    Run,
)

__all__ = [
    "ArtifactUploadError",
    "AuthenticationError",
    "Config",
    "Dataset",
    "Deployment",
    "DeploymentError",
    "EvaluationFailedError",
    "EvaluationJob",
    "LocalMLError",
    "ModelRegistrationError",
    "ModelVersion",
    "PredictionFailedError",
    "PredictionJob",
    "PromptVersion",
    "Run",
    "ValidationError",
    "configure",
    "datasets",
    "deploy",
    "evals",
    "evaluate",
    "huggingface",
    "jax",
    "log_artifact",
    "log_metrics",
    "log_params",
    "mlx",
    "predict",
    "prompts",
    "register_model",
    "start_run",
    "torch",
]

__version__ = "0.1.0"
