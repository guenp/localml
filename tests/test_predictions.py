"""Prediction jobs: providers, the worker loop, and the API endpoints.

The loop tests drive :func:`app.prediction.run_prediction_job` directly against a seeded
SQLite database with a stub provider — the same code path the worker and the inline
fallback execute. API tests cover resolution, pre-flight validation, idempotency, and
result retrieval semantics.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from app.db import Base, Dataset, Model, ModelVersion, PredictionJob, Project, PromptVersion
from app.inference import (
    EchoProvider,
    InferenceError,
    InferenceResult,
    OpenAICompatibleProvider,
    get_provider,
)
from app.prediction import results_path, run_prediction_job
from app.templating import extract_variables
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker

# -- providers -------------------------------------------------------------------


def test_echo_provider_is_deterministic():
    result = EchoProvider().generate("Q: hi\nA:", {})
    assert result.output == "Q: hi\nA:"
    assert result.prompt_tokens == result.completion_tokens == 3


def _openai_provider(handler) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://backend:11434",
        model="tiny",
        transport=httpx.MockTransport(handler),
    )


def test_openai_provider_parses_chat_completion():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "42"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    result = _openai_provider(handler).generate("meaning of life?", {"temperature": 0.2})
    assert result.output == "42"
    assert (result.prompt_tokens, result.completion_tokens) == (7, 1)
    assert result.latency_ms >= 0
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "tiny"
    assert seen["body"]["temperature"] == 0.2
    assert seen["body"]["messages"] == [{"role": "user", "content": "meaning of life?"}]


def test_openai_provider_raises_on_http_error():
    provider = _openai_provider(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(InferenceError, match="HTTP 500"):
        provider.generate("hi", {})


def test_openai_provider_raises_on_malformed_body():
    provider = _openai_provider(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(InferenceError, match="malformed"):
        provider.generate("hi", {})


def test_get_provider_rejects_unknown_name():
    with pytest.raises(InferenceError, match="unknown inference provider"):
        get_provider("bespoke", model="m", config={})


# -- the prediction loop -----------------------------------------------------------


class StubProvider:
    """Records calls; fails for prompts containing any of ``fail_for``."""

    def __init__(self, fail_for: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.fail_for = fail_for

    def generate(self, prompt: str, config: dict[str, Any]) -> InferenceResult:
        self.calls.append(prompt)
        if any(marker in prompt for marker in self.fail_for):
            raise RuntimeError("backend down")
        return InferenceResult(output=f"out({prompt})", latency_ms=1.0)


@pytest.fixture
def db_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    engine.dispose()


def _seed_job(
    db_factory,
    tmp_path,
    rows: list[dict[str, Any]],
    *,
    template: str = "Q: {question}\nA:",
    provider: str = "echo",
    config: dict[str, Any] | None = None,
    artifact_uri: str | None = None,
) -> str:
    if artifact_uri is None:
        dataset_file = tmp_path / "rows.jsonl"
        dataset_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
        artifact_uri = str(dataset_file)
    db = db_factory()
    project = Project(name="local")
    db.add(project)
    db.flush()
    model = Model(project_id=project.id, name="assistant")
    db.add(model)
    db.flush()
    mv = ModelVersion(
        model_id=model.id, version=1, framework="mlx", artifact_uri="file:///m", status="created"
    )
    dataset = Dataset(
        project_id=project.id,
        name="evalset",
        version="v1",
        artifact_uri=artifact_uri,
        row_count=len(rows),
        example_ids=[f"e{i}" for i in range(len(rows))],
    )
    prompt = PromptVersion(
        project_id=project.id,
        name="qa",
        version="v1",
        template=template,
        variables=extract_variables(template),
    )
    db.add_all([mv, dataset, prompt])
    db.flush()
    job = PredictionJob(
        model_version_id=mv.id,
        dataset_id=dataset.id,
        prompt_version_id=prompt.id,
        status="queued",
        provider=provider,
        config=config or {},
    )
    db.add(job)
    db.commit()
    job_id = job.id
    db.close()
    return job_id


def _get_job(db_factory, job_id: str) -> PredictionJob:
    db = db_factory()
    try:
        return db.get(PredictionJob, job_id)
    finally:
        db.close()


def test_loop_completes_and_writes_results(db_factory, tmp_path):
    rows = [{"question": "a", "expected": "a"}, {"question": "b", "expected": "b"}]
    job_id = _seed_job(db_factory, tmp_path, rows, config={"batch_size": 2})

    run_prediction_job(db_factory, job_id)

    job = _get_job(db_factory, job_id)
    assert job.status == "completed"
    assert job.completed_examples == ["e0", "e1"]
    assert job.summary["total"] == 2
    assert job.summary["succeeded"] == 2
    assert job.summary["errored"] == 0
    assert job.results_uri == str(results_path(job_id))

    records = [json.loads(line) for line in results_path(job_id).read_text().splitlines()]
    assert [r["example_id"] for r in records] == ["e0", "e1"]
    # The echo provider returns the rendered prompt, and the full input row is preserved.
    assert records[0]["rendered_prompt"] == records[0]["output"] == "Q: a\nA:"
    assert records[0]["input"] == rows[0]
    assert records[0]["error"] is None


def test_loop_emits_error_rows_without_failing_the_job(db_factory, tmp_path):
    rows = [{"question": "ok"}, {"other": "no question key"}, {"question": "fails"}]
    job_id = _seed_job(db_factory, tmp_path, rows)
    provider = StubProvider(fail_for=("fails",))

    run_prediction_job(db_factory, job_id, provider=provider)

    job = _get_job(db_factory, job_id)
    assert job.status == "completed"
    assert job.summary == {**job.summary, "total": 3, "succeeded": 1, "errored": 2}

    records = {
        r["example_id"]: r for r in map(json.loads, results_path(job_id).read_text().splitlines())
    }
    assert records["e0"]["error"] is None
    assert "missing variables: question" in records["e1"]["error"]
    assert "backend down" in records["e2"]["error"]
    # The row with a missing variable never reached the provider.
    assert provider.calls == ["Q: ok\nA:", "Q: fails\nA:"]


def test_loop_resumes_skipping_completed_examples(db_factory, tmp_path):
    rows = [{"question": "a"}, {"question": "b"}, {"question": "c"}]
    job_id = _seed_job(db_factory, tmp_path, rows)

    # Simulate a prior partial run: e0 already inferred and checkpointed.
    db = db_factory()
    job = db.get(PredictionJob, job_id)
    job.completed_examples = ["e0"]
    db.commit()
    db.close()
    path = results_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"example_id": "e0", "output": "prior", "error": None}) + "\n")

    provider = StubProvider()
    run_prediction_job(db_factory, job_id, provider=provider)

    # Only the two pending examples were inferred; the prior record is kept.
    assert provider.calls == ["Q: b\nA:", "Q: c\nA:"]
    job = _get_job(db_factory, job_id)
    assert job.status == "completed"
    assert job.completed_examples == ["e0", "e1", "e2"]
    assert job.summary["total"] == 3


def test_loop_is_a_noop_on_terminal_jobs(db_factory, tmp_path):
    job_id = _seed_job(db_factory, tmp_path, [{"question": "a"}])
    db = db_factory()
    db.get(PredictionJob, job_id).status = "completed"
    db.commit()
    db.close()

    provider = StubProvider()
    run_prediction_job(db_factory, job_id, provider=provider)
    assert provider.calls == []
    assert _get_job(db_factory, job_id).status == "completed"


def test_loop_fails_job_when_dataset_unreadable(db_factory, tmp_path):
    job_id = _seed_job(
        db_factory, tmp_path, [{"question": "a"}], artifact_uri=str(tmp_path / "missing.jsonl")
    )
    run_prediction_job(db_factory, job_id)
    job = _get_job(db_factory, job_id)
    assert job.status == "failed"
    assert "dataset rows unavailable" in job.error
    assert job.completed_at is not None


# -- API endpoints ------------------------------------------------------------------


def _setup_triple(client, *, rows=None, template="Q: {question}\nA:") -> None:
    rows = rows if rows is not None else [{"question": "a", "expected": "a"}]
    assert (
        client.post(
            "/models/m/versions",
            json={
                "model_name": "m",
                "framework": "mlx",
                "artifact_uri": "file:///m",
            },
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/datasets",
            json={
                "project": "local",
                "name": "evalset",
                "artifact_uri": "/tmp/rows.jsonl",
                "rows": rows,
            },
        ).status_code
        == 201
    )
    assert client.post("/prompts", json={"name": "qa", "template": template}).status_code == 201


def _create_prediction(client, **overrides):
    body = {"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "echo"}
    body.update(overrides)
    return client.post("/predictions", json=body)


def test_create_prediction_resolves_references(client):
    _setup_triple(client)
    resp = _create_prediction(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["provider"] == "echo"
    assert body["total_count"] == 1
    assert body["completed_count"] == 0
    # The stored ids are canonical, not the name:version references.
    assert ":" not in body["model_version_id"]

    got = client.get(f"/predictions/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body


def test_create_prediction_preflights_prompt_variables(client):
    _setup_triple(client, template="Q: {question}\nContext: {context}\nA:")
    resp = _create_prediction(client)
    assert resp.status_code == 422
    assert "context" in resp.json()["detail"]


def test_create_prediction_skips_preflight_without_columns(client):
    # Rows not provided at registration -> columns unknown -> validated per-row at run time.
    _setup_triple(client, rows=[], template="Q: {question}\nA:")
    assert _create_prediction(client).status_code == 201


def test_create_prediction_404_on_unknown_references(client):
    _setup_triple(client)
    assert _create_prediction(client, model="ghost:v1").status_code == 404
    assert _create_prediction(client, dataset="ghost:v1").status_code == 404
    assert _create_prediction(client, prompt="ghost:v1").status_code == 404


def test_create_prediction_idempotency(client):
    _setup_triple(client)
    headers = {"Idempotency-Key": "pred-1"}
    first = _create_prediction(client)
    resp1 = client.post(
        "/predictions",
        json={"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "echo"},
        headers=headers,
    )
    resp2 = client.post(
        "/predictions",
        json={"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "echo"},
        headers=headers,
    )
    assert resp1.json()["id"] == resp2.json()["id"] != first.json()["id"]

    mismatch = client.post(
        "/predictions",
        json={"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "openai"},
        headers=headers,
    )
    assert mismatch.status_code == 409


def test_get_prediction_404(client):
    assert client.get("/predictions/nope").status_code == 404
    assert client.get("/predictions/nope/results").status_code == 404


def test_results_conflict_before_job_ran(client):
    _setup_triple(client)
    job_id = _create_prediction(client).json()["id"]
    resp = client.get(f"/predictions/{job_id}/results")
    assert resp.status_code == 409
    assert "queued" in resp.json()["detail"]


def test_dataset_registration_captures_columns(client):
    resp = client.post(
        "/datasets",
        json={
            "project": "local",
            "name": "cols",
            "artifact_uri": "/tmp/cols.jsonl",
            "rows": [
                {"question": "a", "expected": "x", "note": "only here"},
                {"question": "b", "expected": "y"},
            ],
        },
    )
    assert resp.status_code == 201
    # Intersection across rows: 'note' is not guaranteed by every row.
    assert resp.json()["columns"] == ["expected", "question"]
