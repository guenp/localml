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


def _predicted_triple(tmp_path, *, name: str):
    """Register a model/dataset/prompt triple and run an echo prediction to completion."""
    import json

    # The echo provider returns the rendered prompt, so the first row's expected matches.
    rows = [{"question": "a", "expected": "Q: a\nA:"}, {"question": "b", "expected": "nope"}]
    dataset_file = tmp_path / f"{name}.jsonl"
    dataset_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset = ml.datasets.register(
        project="local", name=f"{name}-ds", artifact_uri=str(dataset_file), rows=rows
    )
    prompt = ml.prompts.register(name=f"{name}-qa", template="Q: {question}\nA:")
    version = ml.register_model(f"{name}-m", artifact_uri="file:///tmp/m", framework="mlx")
    job = ml.predict(model=version, dataset=dataset, prompt=prompt, provider="echo")
    job.wait(timeout=30)
    return version, dataset, prompt, job


def test_evals_end_to_end(sdk, tmp_path, monkeypatch):
    """Full Phase 3 M3 loop: score a completed prediction, mixing built-in + custom metrics."""
    _, _, _, prediction = _predicted_triple(tmp_path, name="evals-e2e")

    # Custom metrics run client-side over the stored results and persist with the job.
    monkeypatch.setitem(
        ml.evals._LOCAL_METRICS,
        "half_of_total",
        lambda records, config: len(records) / 2,
    )
    job = ml.evals.run(prediction, ["exact_match", "error_rate", "half_of_total"])
    job.wait(timeout=30)

    assert job.status == "completed"
    assert job.prediction_job_id == prediction.id
    assert job.metrics == {"exact_match": 0.5, "error_rate": 0.0, "half_of_total": 1.0}
    assert job.report_uri
    assert job.error is None


def test_evals_run_rejects_unknown_prediction(sdk):
    with pytest.raises(LocalMLError, match="not found"):
        ml.evals.run("ghost-prediction", ["error_rate"])


def test_evaluate_predict_then_eval_sugar(sdk, tmp_path):
    """`ml.evaluate(..., prompt=...)` is predict-then-eval sugar over stored results."""
    import json

    rows = [{"question": "a", "expected": "Q: a\nA:"}]
    dataset_file = tmp_path / "sugar.jsonl"
    dataset_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset = ml.datasets.register(
        project="local", name="sugar-ds", artifact_uri=str(dataset_file), rows=rows
    )
    prompt = ml.prompts.register(name="sugar-qa", template="Q: {question}\nA:")
    version = ml.register_model("sugar-m", artifact_uri="file:///tmp/m", framework="mlx")

    job = ml.evaluate(
        version, dataset, ["exact_match"], prompt=prompt, provider="echo", predict_timeout=30
    )
    assert job.prediction_job_id is not None
    job.wait(timeout=30)
    assert job.metrics == {"exact_match": 1.0}


def test_compare_end_to_end(sdk, tmp_path):
    """Two prompt variants over the same dataset, compared across aligned example ids."""
    import json

    rows = [{"question": "a"}, {"question": "b"}]
    dataset_file = tmp_path / "cmp.jsonl"
    dataset_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset = ml.datasets.register(
        project="local", name="cmp-ds", artifact_uri=str(dataset_file), rows=rows
    )
    p1 = ml.prompts.register(name="cmp-qa", template="Q: {question}\nA:")
    p2 = ml.prompts.register(name="cmp-qa", template="Question: {question}\nAnswer:")
    version = ml.register_model("cmp-m", artifact_uri="file:///tmp/m", framework="mlx")

    job_a = ml.predict(model=version, dataset=dataset, prompt=p1, provider="echo").wait(timeout=30)
    job_b = ml.predict(model=version, dataset=dataset, prompt=p2, provider="echo").wait(timeout=30)

    report = ml.compare(job_a, job_b, max_examples=1)
    assert report.kind == "prediction"
    assert report.differs == ["prompt_version"]
    assert report.rows["aligned"] == 2
    assert report.rows["agreements"] == 0  # every rendered prompt (echo output) changed
    assert len(report.changed_examples) == 1  # capped


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


def _deployable_version(sdk, name: str):
    version = ml.register_model(name, artifact_uri="file:///tmp/m", framework="mlx")
    for target in ("candidate", "staging"):
        sdk._request("POST", f"/models/{name}/versions/1/promote", json={"target_status": target})
    return version


def test_deploy_and_chat_round_trip(sdk, monkeypatch):
    """Deploy through the proxy, register a custom provider, and round-trip a chat request."""
    import json as _json

    import httpx
    from app import serving

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "pong"}}]})

    # The live server runs in-process, so patching the module object is visible to it.
    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(handler))

    version = _deployable_version(sdk, "served")
    ml.providers.register("my-ollama", base_url="http://gpu-box:11434", model="llama3")
    dep = ml.deploy(version, target="local", provider="my-ollama")
    assert dep.config["base_url"] == "http://gpu-box:11434"
    assert dep.status == "active"  # health check is stubbed True in tests

    reply = dep.chat([{"role": "user", "content": "ping"}])
    assert reply["choices"][0]["message"]["content"] == "pong"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["model"] == "llama3"  # from the registered provider

    # predict() sugar wraps the prompt.
    dep.predict({"prompt": "hi"})
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_deployment_hot_swap(sdk, monkeypatch):
    import httpx
    from app import serving

    monkeypatch.setattr(
        serving, "_transport", httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}))
    )
    v1 = _deployable_version(sdk, "swap-a")
    v2 = _deployable_version(sdk, "swap-b")
    dep = ml.deploy(v1, target="local", config={"model": "a"})

    dep.swap(model=v2, config={"model": "b"})
    assert dep.model_version_id == v2.id
    assert dep.config["model"] == "b"
    assert dep.status == "active"


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
