"""SDK↔API integration tests.

These drive the real ``localml`` SDK (HTTPX client, retries, payload construction, error
mapping) against the control plane running under a live uvicorn server. They are the contract
between the SDK and the API: if a payload or route drifts, these fail.
"""

from __future__ import annotations

import pytest

import localml as ml
from localml.exceptions import LocalMLError


def test_run_context_manager_round_trips(sdk, tmp_path):
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights")

    with ml.start_run(project="local", config={"lr": 1e-3}) as run:
        ml.log_metrics({"accuracy": 0.9})
        ml.log_params({"batch_size": 8})
        ml.log_artifact(str(artifact), artifact_type="model")
        assert run.status == "running"

    # On clean exit the run is marked completed, and the server agrees.
    assert run.status == "completed"
    assert sdk.get_run(run.id).status == "completed"


def test_run_marked_failed_on_exception(sdk):
    with pytest.raises(ValueError), ml.start_run(project="local", config={}) as run:
        raise ValueError("boom")
    assert run.status == "failed"


def test_register_model_and_fetch(sdk):
    version = ml.register_model(
        "assistant",
        artifact_uri="file:///tmp/assistant",
        framework="mlx",
        metadata={"task": "chat"},
    )
    assert version.version == 1
    assert version.status == "created"
    # A second registration bumps the version.
    v2 = ml.register_model("assistant", artifact_uri="file:///tmp/assistant", framework="mlx")
    assert v2.version == 2


def test_dataset_register_and_get(sdk):
    ds = ml.datasets.register(
        project="local",
        name="evalset",
        artifact_uri="s3://localml-artifacts/datasets/evalset.jsonl",
        rows=[{"prompt": "a", "expected": "b"}, {"example_id": "known", "prompt": "c"}],
    )
    assert ds.version == "v1"
    assert ds.row_count == 2
    assert ds.example_ids[1] == "known"

    versions = ml.datasets.get("evalset")
    assert [d.version for d in versions] == ["v1"]


def test_evaluate_queues_job(sdk):
    version = ml.register_model("m", artifact_uri="file:///tmp/m", framework="mlx")
    job = ml.evaluate(model=version, dataset="evalset:v1", metrics=["accuracy"])
    assert job.status == "queued"
    # The job handle can refresh from the server.
    assert job.refresh().model_version_id == version.id


def test_deploy_rejects_non_deployable_model(sdk):
    version = ml.register_model("m", artifact_uri="file:///tmp/m", framework="mlx")
    # A freshly-created version is not deployable -> API 409 -> SDK error.
    with pytest.raises(LocalMLError):
        ml.deploy(model=version, target="local")


def test_client_create_and_get_run(sdk):
    r1 = sdk.create_run(project="local", config={"seed": 1})
    assert sdk.get_run(r1.id).id == r1.id
