"""Evaluation jobs: metric registry, scoring loop, and API endpoints (Phase 3 M3)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from app import evaluation
from app.db import (
    Base,
    Dataset,
    EvaluationJob,
    EvaluationMetric,
    Model,
    ModelVersion,
    PredictionJob,
    Project,
    PromptVersion,
)
from app.evaluation import (
    metric_names,
    register_metric,
    report_path,
    run_evaluation_job,
    validate_metrics,
)
from app.prediction import results_path, run_prediction_job
from app.session import get_db
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker

# -- metric registry ----------------------------------------------------------------


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "example_id": "e0",
        "input": {"question": "q", "expected": "yes"},
        "rendered_prompt": "p",
        "output": "yes",
        "latency_ms": 10.0,
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "error": None,
    }
    base.update(overrides)
    return base


def _metric(name: str, records: list[dict[str, Any]], config: dict[str, Any] | None = None):
    return evaluation._METRICS[name](records, config or {})


def test_builtin_metrics_are_registered():
    expected = {
        "exact_match",
        "contains_expected",
        "regex_match",
        "format_validity",
        "json_validity",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "error_rate",
        "avg_input_tokens",
        "avg_output_tokens",
    }
    assert expected <= set(metric_names())


def test_exact_match_and_contains_expected():
    records = [
        _record(output="yes"),
        _record(output=" yes \n"),  # normalized
        _record(output="well, yes indeed"),
        _record(output="no"),
    ]
    assert _metric("exact_match", records) == 0.5
    assert _metric("contains_expected", records) == 0.75


def test_quality_metrics_score_around_errors_and_missing_expected():
    records = [
        _record(output="yes"),
        _record(output=None, error="backend down"),  # errored -> excluded
        _record(input={"question": "q"}, output="yes"),  # no expected -> excluded
    ]
    assert _metric("exact_match", records) == 1.0
    # Nothing scorable at all -> None (reported as skipped, not 0.0).
    assert _metric("exact_match", [_record(input={"question": "q"})]) is None
    # A different expected column can be configured.
    records = [_record(input={"question": "q", "label": "yes"}, output="yes")]
    assert _metric("exact_match", records, {"expected_field": "label"}) == 1.0


def test_regex_and_format_and_json_validity():
    records = [
        _record(output='{"a": 1}'),
        _record(output="ANSWER: 42"),
        _record(output="   "),
    ]
    assert _metric("regex_match", records, {"pattern": r"ANSWER: \d+"}) == pytest.approx(1 / 3)
    assert _metric("json_validity", records) == pytest.approx(1 / 3)
    assert _metric("format_validity", records) == pytest.approx(2 / 3)  # non-empty text
    assert _metric("format_validity", records, {"format": "json"}) == pytest.approx(1 / 3)


def test_latency_percentiles_and_token_averages():
    records = [_record(latency_ms=v) for v in (10.0, 20.0, 30.0, 40.0)]
    assert _metric("latency_p50", records) == 25.0
    assert _metric("latency_p99", records) == pytest.approx(39.7)
    assert _metric("latency_p50", [_record(latency_ms=None)]) is None
    assert _metric("avg_input_tokens", [_record(prompt_tokens=4), _record(prompt_tokens=8)]) == 6.0
    assert _metric("avg_output_tokens", [_record(completion_tokens=None)]) is None


def test_error_rate():
    records = [_record(), _record(error="boom"), _record(error="boom"), _record()]
    assert _metric("error_rate", records) == 0.5
    assert _metric("error_rate", []) is None


def test_validate_metrics_reports_problems():
    assert validate_metrics(["exact_match"], {}) == []
    assert validate_metrics(["nope"], {}) == ["unknown metric: nope"]
    assert validate_metrics(["regex_match"], {}) == ["regex_match requires config['pattern']"]
    assert validate_metrics(["regex_match"], {"pattern": "x"}) == []


def test_register_metric_custom(monkeypatch):
    monkeypatch.setitem(evaluation._METRICS, "always_one", lambda records, config: 1.0)
    assert validate_metrics(["always_one"], {}) == []
    with pytest.raises(TypeError):
        register_metric("bad", "not callable")  # type: ignore[arg-type]


# -- scoring loop -------------------------------------------------------------------


@pytest.fixture
def db_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    engine.dispose()


def _seed_eval(
    db_factory,
    records: list[dict[str, Any]] | None,
    *,
    metrics: list[str],
    config: dict[str, Any] | None = None,
    prediction_status: str = "completed",
    link: bool = True,
    client_metrics: dict[str, float] | None = None,
) -> str:
    """Seed a prediction job (with stored results) and a queued evaluation over it."""
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
        project_id=project.id, name="evalset", version="v1", artifact_uri="/x", row_count=0
    )
    prompt = PromptVersion(
        project_id=project.id,
        name="qa",
        version="v1",
        template="{question}",
        variables=["question"],
    )
    db.add_all([mv, dataset, prompt])
    db.flush()
    prediction = PredictionJob(
        model_version_id=mv.id,
        dataset_id=dataset.id,
        prompt_version_id=prompt.id,
        status=prediction_status,
        provider="echo",
    )
    db.add(prediction)
    db.flush()
    if records is not None:
        path = results_path(prediction.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    job = EvaluationJob(
        prediction_job_id=prediction.id if link else None,
        model_version_id=mv.id,
        dataset_id=dataset.id,
        status="queued",
        config={"metrics": metrics, **(config or {})},
    )
    db.add(job)
    db.flush()
    for name, value in (client_metrics or {}).items():
        db.add(EvaluationMetric(evaluation_job_id=job.id, name=name, value=value))
    db.commit()
    job_id = job.id
    db.close()
    return job_id


def _get_eval(db_factory, job_id: str) -> EvaluationJob:
    db = db_factory()
    try:
        job = db.get(EvaluationJob, job_id)
        _ = list(job.metrics_rows)  # load the relationship before the session closes
        return job
    finally:
        db.close()


def test_eval_completes_with_metric_rows_and_report(db_factory):
    records = [
        _record(example_id="e0", output="yes", latency_ms=10.0),
        _record(example_id="e1", output="no", latency_ms=30.0),
        _record(example_id="e2", output=None, latency_ms=None, error="backend down"),
    ]
    job_id = _seed_eval(db_factory, records, metrics=["exact_match", "error_rate", "latency_p50"])

    run_evaluation_job(db_factory, job_id)

    job = _get_eval(db_factory, job_id)
    assert job.status == "completed"
    assert job.error is None
    assert job.completed_at is not None
    metrics = {m.name: m.value for m in job.metrics_rows}
    assert metrics == {
        "exact_match": 0.5,
        "error_rate": pytest.approx(1 / 3),
        "latency_p50": 20.0,
    }

    assert job.report_uri == str(report_path(job_id))
    report = json.loads(report_path(job_id).read_text())
    assert report["evaluation_job_id"] == job_id
    assert report["prediction_job_id"] == job.prediction_job_id
    assert report["metrics"] == {**metrics, "error_rate": report["metrics"]["error_rate"]}
    assert report["counts"] == {"total": 3, "succeeded": 2, "errored": 1}
    assert report["skipped"] == {}


def test_eval_records_unscorable_and_erroring_metrics_as_skipped(db_factory, monkeypatch):
    def boom(records, config):
        raise ValueError("bad metric config")

    monkeypatch.setitem(evaluation._METRICS, "boom", boom)
    records = [_record(input={"question": "q"}, output="yes")]  # no expected field
    job_id = _seed_eval(db_factory, records, metrics=["exact_match", "boom", "error_rate"])

    run_evaluation_job(db_factory, job_id)

    job = _get_eval(db_factory, job_id)
    assert job.status == "completed"
    assert {m.name for m in job.metrics_rows} == {"error_rate"}
    report = json.loads(report_path(job_id).read_text())
    assert report["skipped"]["exact_match"] == "no scorable records"
    assert report["skipped"]["boom"] == "ValueError: bad metric config"


def test_eval_keeps_client_metric_rows(db_factory):
    records = [_record(output="yes")]
    job_id = _seed_eval(
        db_factory,
        records,
        metrics=["exact_match"],
        client_metrics={"my_metric": 0.5, "exact_match": 9.9},  # pre-existing row wins
    )

    run_evaluation_job(db_factory, job_id)

    job = _get_eval(db_factory, job_id)
    metrics = {m.name: m.value for m in job.metrics_rows}
    assert metrics == {"my_metric": 0.5, "exact_match": 9.9}
    assert len(job.metrics_rows) == 2  # no duplicate exact_match row
    report = json.loads(report_path(job_id).read_text())
    assert report["metrics"] == metrics


def test_eval_fails_legacy_job_without_prediction_link(db_factory):
    job_id = _seed_eval(db_factory, None, metrics=["exact_match"], link=False)
    run_evaluation_job(db_factory, job_id)
    job = _get_eval(db_factory, job_id)
    assert job.status == "failed"
    assert "no prediction link" in job.error
    assert job.error.startswith("Traceback")  # traceback captured
    assert job.completed_at is not None


def test_eval_fails_when_prediction_incomplete(db_factory):
    job_id = _seed_eval(db_factory, None, metrics=["exact_match"], prediction_status="running")
    run_evaluation_job(db_factory, job_id)
    job = _get_eval(db_factory, job_id)
    assert job.status == "failed"
    assert "is not completed" in job.error


def test_eval_fails_when_results_missing(db_factory):
    job_id = _seed_eval(db_factory, None, metrics=["exact_match"])
    run_evaluation_job(db_factory, job_id)
    job = _get_eval(db_factory, job_id)
    assert job.status == "failed"
    assert "results unavailable" in job.error


def test_eval_retries_transient_failures_with_bound(db_factory, monkeypatch):
    calls = []

    def flaky(db, job):
        calls.append(1)
        raise RuntimeError("transient db hiccup")

    monkeypatch.setattr(evaluation, "_run", flaky)
    job_id = _seed_eval(db_factory, None, metrics=["error_rate"], config={"max_attempts": 2})

    run_evaluation_job(db_factory, job_id)

    assert len(calls) == 2  # bounded retries
    job = _get_eval(db_factory, job_id)
    assert job.status == "failed"
    assert "transient db hiccup" in job.error


def test_eval_does_not_retry_deterministic_failures(db_factory, monkeypatch):
    calls = []
    original = evaluation._run

    def once(db, job):
        calls.append(1)
        return original(db, job)

    monkeypatch.setattr(evaluation, "_run", once)
    job_id = _seed_eval(db_factory, None, metrics=["error_rate"], link=False)
    run_evaluation_job(db_factory, job_id)
    assert len(calls) == 1
    assert _get_eval(db_factory, job_id).status == "failed"


def test_eval_is_a_noop_on_terminal_jobs(db_factory):
    job_id = _seed_eval(db_factory, None, metrics=["error_rate"])
    db = db_factory()
    db.get(EvaluationJob, job_id).status = "completed"
    db.commit()
    db.close()
    run_evaluation_job(db_factory, job_id)
    assert _get_eval(db_factory, job_id).status == "completed"
    assert _get_eval(db_factory, job_id).error is None


# -- API endpoints ------------------------------------------------------------------


def _session_factory(client):
    override = client.app.dependency_overrides[get_db]
    return lambda: next(override())


def _completed_prediction(client, tmp_path) -> str:
    """Register a model/dataset/prompt triple, run a prediction to completion, return its id."""
    rows = [
        {"question": "a", "expected": "Q: a\nA:"},  # echo output == rendered prompt
        {"question": "b", "expected": "nope"},
    ]
    rows_file = tmp_path / "rows.jsonl"
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert (
        client.post(
            "/models/m/versions",
            json={"model_name": "m", "framework": "mlx", "artifact_uri": "file:///m"},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/datasets",
            json={
                "project": "local",
                "name": "evalset",
                "artifact_uri": str(rows_file),
                "rows": rows,
            },
        ).status_code
        == 201
    )
    assert (
        client.post("/prompts", json={"name": "qa", "template": "Q: {question}\nA:"}).status_code
        == 201
    )
    resp = client.post(
        "/predictions",
        json={"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "echo"},
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]
    run_prediction_job(_session_factory(client), job_id)
    assert client.get(f"/predictions/{job_id}").json()["status"] == "completed"
    return job_id


def test_create_and_run_evaluation_via_api(client, tmp_path):
    prediction_id = _completed_prediction(client, tmp_path)
    resp = client.post(
        "/evaluations",
        json={"prediction": prediction_id, "metrics": ["exact_match", "error_rate"]},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["prediction_job_id"] == prediction_id
    assert body["model_version_id"]  # denormalized from the prediction
    assert body["metrics"] is None

    run_evaluation_job(_session_factory(client), body["id"])

    done = client.get(f"/evaluations/{body['id']}").json()
    assert done["status"] == "completed"
    assert done["metrics"] == {"exact_match": 0.5, "error_rate": 0.0}
    assert done["report_uri"]


def test_create_evaluation_conflicts_on_incomplete_prediction(client, tmp_path):
    _completed_prediction(client, tmp_path)
    # A second prediction that stays queued (inline jobs are off in unit tests).
    queued = client.post(
        "/predictions",
        json={"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "echo"},
    ).json()
    resp = client.post("/evaluations", json={"prediction": queued["id"], "metrics": ["error_rate"]})
    assert resp.status_code == 409
    assert "not completed" in resp.json()["detail"]


def test_create_evaluation_validation_errors(client, tmp_path):
    prediction_id = _completed_prediction(client, tmp_path)
    cases = [
        ({"prediction": "ghost", "metrics": ["error_rate"]}, 404, "not found"),
        ({"prediction": prediction_id, "metrics": []}, 422, "at least one metric"),
        ({"prediction": prediction_id, "metrics": ["nope"]}, 422, "unknown metric"),
        ({"prediction": prediction_id, "metrics": ["regex_match"]}, 422, "pattern"),
        ({"metrics": ["error_rate"]}, 422, "exactly one"),
        (
            {"prediction": prediction_id, "model_version_id": "x", "metrics": ["error_rate"]},
            422,
            "exactly one",
        ),
    ]
    for body, code, needle in cases:
        resp = client.post("/evaluations", json=body)
        assert resp.status_code == code, body
        assert needle in resp.json()["detail"]


def test_create_evaluation_persists_client_metrics(client, tmp_path):
    prediction_id = _completed_prediction(client, tmp_path)
    resp = client.post(
        "/evaluations",
        json={"prediction": prediction_id, "metrics": [], "client_metrics": {"my_metric": 0.5}},
    )
    assert resp.status_code == 201
    assert resp.json()["metrics"] == {"my_metric": 0.5}


def test_create_evaluation_idempotent_replay(client, tmp_path):
    prediction_id = _completed_prediction(client, tmp_path)
    body = {"prediction": prediction_id, "metrics": ["error_rate"]}
    headers = {"Idempotency-Key": "eval-m3-1"}
    first = client.post("/evaluations", json=body, headers=headers).json()
    replay = client.post("/evaluations", json=body, headers=headers).json()
    assert first["id"] == replay["id"]
    mismatch = client.post(
        "/evaluations", json={**body, "metrics": ["exact_match"]}, headers=headers
    )
    assert mismatch.status_code == 409
