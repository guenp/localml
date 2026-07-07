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


def test_prompt_register_render_and_get(sdk):
    from localml.exceptions import ValidationError

    prompt = ml.prompts.register(name="qa", template="Q: {question}\nA:")
    assert prompt.version == "v1"
    assert prompt.variables == ["question"]

    # The handle renders server-side, and mismatched variables map to the SDK error type.
    assert prompt.render(question="why?") == "Q: why?\nA:"
    with pytest.raises(ValidationError):
        prompt.render(question="why?", bogus=1)

    v2 = ml.prompts.register(name="qa", template="Q: {question}\nContext: {context}\nA:")
    assert v2.version == "v2"
    assert [p.version for p in ml.prompts.get("qa")] == ["v1", "v2"]
    assert ml.prompts.render("qa", "v2", question="q", context="c") == "Q: q\nContext: c\nA:"


def test_predict_end_to_end(sdk, tmp_path):
    """Full Phase 3 M2 loop: register the triple, predict with the echo provider, read results.

    No Redis in tests, so the job runs on the API's inline background thread — the same
    degradation path the standalone (no-Docker) flow uses.
    """
    import json

    rows = [{"question": "a", "expected": "a"}, {"question": "b", "expected": "b"}]
    dataset_file = tmp_path / "eval.jsonl"
    dataset_file.write_text("".join(json.dumps(r) + "\n" for r in rows))

    dataset = ml.datasets.register(
        project="local", name="pred-eval", artifact_uri=str(dataset_file), rows=rows
    )
    prompt = ml.prompts.register(name="pred-qa", template="Q: {question}\nA:")
    version = ml.register_model("pred-m", artifact_uri="file:///tmp/m", framework="mlx")

    job = ml.predict(
        model=version, dataset=dataset, prompt=prompt, provider="echo", config={"batch_size": 2}
    )
    assert job.status == "queued"
    job.wait(timeout=30)

    assert job.status == "completed"
    assert (job.completed_count, job.total_count) == (2, 2)
    assert job.summary["succeeded"] == 2

    results = job.results()
    assert len(results) == 2
    first = next(r for r in results if r["input"]["question"] == "a")
    assert first["rendered_prompt"] == first["output"] == "Q: a\nA:"
    assert first["error"] is None


def test_predict_rejects_prompt_needing_missing_column(sdk, tmp_path):
    from localml.exceptions import ValidationError

    dataset = ml.datasets.register(
        project="local",
        name="pred-narrow",
        artifact_uri=str(tmp_path / "narrow.jsonl"),
        rows=[{"question": "a"}],
    )
    prompt = ml.prompts.register(name="pred-ctx", template="Q: {question}\nContext: {context}")
    version = ml.register_model("pred-m2", artifact_uri="file:///tmp/m", framework="mlx")

    with pytest.raises(ValidationError, match="context"):
        ml.predict(model=version, dataset=dataset, prompt=prompt, provider="echo")


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


def test_log_artifact_records_checksum(sdk, tmp_path):
    from localml._hashing import sha256_file

    run = sdk.create_run(project="local", config={})
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"hello world")
    record = sdk.log_artifact(
        run.id, uri=artifact.as_uri(), artifact_type="model", checksum=sha256_file(artifact)
    )
    assert record["checksum"] == sha256_file(artifact)
    # No MinIO in tests, so there is no upload target and nothing to PUT.
    assert record["upload_url"] is None


def test_log_artifact_accepts_directory(sdk, tmp_path):
    d = tmp_path / "ckpt"
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"a")
    (d / "sub" / "b.bin").write_bytes(b"b")

    with ml.start_run(project="local", config={}):
        ml.log_artifact(str(d), artifact_type="dir")
    # The directory was bundled next to itself and recorded without error.
    assert (tmp_path / "ckpt.tar.gz").exists()
