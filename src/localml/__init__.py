"""localml — Local ML experimentation platform SDK.

Public API::

    import localml as ml

    ml.configure(api_url="http://localhost:8000", token="local-dev-token")

    with ml.start_run(project="demo", config={"lr": 0.001}) as run:
        ml.log_metrics({"accuracy": 0.91})
        ml.log_artifact("outputs/model.safetensors")
        version = ml.mlx.log_model(name="assistant", model_dir="./model")
        job = ml.evaluate(model=version, dataset="datasets/eval.jsonl",
                          metrics=["accuracy"])
        job.wait()
        ml.deploy(model=version, target="local")
"""

from __future__ import annotations

from . import huggingface, jax, mlx, torch
from .config import Config, configure
from .exceptions import (
    ArtifactUploadError,
    AuthenticationError,
    DeploymentError,
    EvaluationFailedError,
    LocalMLError,
    ModelRegistrationError,
    ValidationError,
)
from .ops import deploy, evaluate, log_artifact, log_metrics, log_params, register_model
from .run import start_run
from .types import Deployment, EvaluationJob, ModelVersion, Run

__all__ = [
    # configuration
    "Config",
    "configure",
    # run lifecycle
    "start_run",
    "log_metrics",
    "log_params",
    "log_artifact",
    "register_model",
    "evaluate",
    "deploy",
    # types
    "Run",
    "ModelVersion",
    "EvaluationJob",
    "Deployment",
    # framework adapters
    "torch",
    "jax",
    "mlx",
    "huggingface",
    # exceptions
    "LocalMLError",
    "AuthenticationError",
    "ValidationError",
    "ArtifactUploadError",
    "ModelRegistrationError",
    "EvaluationFailedError",
    "DeploymentError",
]

__version__ = "0.1.0"
