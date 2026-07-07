"""Comparison reports across two prediction/evaluation jobs (Phase 3 M4)."""

from __future__ import annotations

import json

from app.evaluation import run_evaluation_job
from app.prediction import run_prediction_job
from app.session import get_db


def _session_factory(client):
    override = client.app.dependency_overrides[get_db]
    return lambda: next(override())


def _setup(client, tmp_path) -> None:
    rows = [
        {"question": "a", "expected": "Q: a\nA:"},
        {"question": "b", "expected": "Q: b\nA:"},
        {"question": "c", "expected": "nope"},
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
    # Two prompt versions: the second renders differently for every row.
    for template in ("Q: {question}\nA:", "Question: {question}\nAnswer:"):
        assert client.post("/prompts", json={"name": "qa", "template": template}).status_code == 201


def _completed_prediction(client, prompt_version: str) -> str:
    resp = client.post(
        "/predictions",
        json={
            "model": "m:v1",
            "dataset": "evalset:v1",
            "prompt": f"qa:{prompt_version}",
            "provider": "echo",
        },
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]
    run_prediction_job(_session_factory(client), job_id)
    return job_id


def test_compare_predictions(client, tmp_path):
    _setup(client, tmp_path)
    a = _completed_prediction(client, "v1")
    b = _completed_prediction(client, "v2")

    resp = client.get("/compare", params={"a": a, "b": b})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "prediction"
    assert body["differs"] == ["prompt_version"]
    assert body["a"]["prediction_job_id"] == a
    assert body["b"]["prediction_job_id"] == b
    assert body["metrics"] == {}  # not evaluations

    rows = body["rows"]
    assert rows["aligned"] == 3
    assert rows["only_in_a"] == rows["only_in_b"] == 0
    assert rows["both_succeeded"] == 3
    assert rows["agreements"] == 0  # every rendered prompt (echo output) changed
    assert rows["a_errored"] == rows["b_errored"] == 0
    assert rows["mean_latency_ms"]["a"] is not None

    changed = body["changed_examples"]
    assert len(changed) == 3
    assert changed[0]["output_a"] != changed[0]["output_b"]


def test_compare_identical_prediction_agrees(client, tmp_path):
    _setup(client, tmp_path)
    a = _completed_prediction(client, "v1")
    resp = client.get("/compare", params={"a": a, "b": a})
    body = resp.json()
    assert body["differs"] == []
    assert body["rows"]["agreements"] == 3
    assert body["changed_examples"] == []


def test_compare_respects_max_examples(client, tmp_path):
    _setup(client, tmp_path)
    a = _completed_prediction(client, "v1")
    b = _completed_prediction(client, "v2")
    body = client.get("/compare", params={"a": a, "b": b, "max_examples": 1}).json()
    assert body["rows"]["aligned"] == 3
    assert len(body["changed_examples"]) == 1


def test_compare_evaluations_includes_metric_deltas(client, tmp_path):
    _setup(client, tmp_path)
    pred_a = _completed_prediction(client, "v1")
    pred_b = _completed_prediction(client, "v2")
    evals = []
    for pred in (pred_a, pred_b):
        resp = client.post(
            "/evaluations", json={"prediction": pred, "metrics": ["exact_match", "error_rate"]}
        )
        assert resp.status_code == 201
        run_evaluation_job(_session_factory(client), resp.json()["id"])
        evals.append(resp.json()["id"])

    body = client.get("/compare", params={"a": evals[0], "b": evals[1]}).json()
    assert body["kind"] == "evaluation"
    assert body["a"]["job_id"] == evals[0]
    assert body["a"]["prediction_job_id"] == pred_a
    # v1's expected values match its rendered prompts for 2/3 rows; v2's for none.
    assert body["metrics"]["exact_match"] == {
        "a": 2 / 3,
        "b": 0.0,
        "delta": -2 / 3,
    }
    assert body["metrics"]["error_rate"]["delta"] == 0.0
    assert body["a"]["metrics"]["exact_match"] == 2 / 3


def test_compare_mixed_references_is_a_prediction_comparison(client, tmp_path):
    _setup(client, tmp_path)
    pred_a = _completed_prediction(client, "v1")
    pred_b = _completed_prediction(client, "v2")
    ev = client.post("/evaluations", json={"prediction": pred_b, "metrics": ["error_rate"]}).json()
    body = client.get("/compare", params={"a": pred_a, "b": ev["id"]}).json()
    assert body["kind"] == "prediction"
    assert body["b"]["prediction_job_id"] == pred_b
    assert body["metrics"] == {}


def test_compare_error_cases(client, tmp_path):
    _setup(client, tmp_path)
    a = _completed_prediction(client, "v1")

    assert client.get("/compare", params={"a": a, "b": "ghost"}).status_code == 404

    # An incomplete prediction cannot be compared.
    queued = client.post(
        "/predictions",
        json={"model": "m:v1", "dataset": "evalset:v1", "prompt": "qa:v1", "provider": "echo"},
    ).json()
    resp = client.get("/compare", params={"a": a, "b": queued["id"]})
    assert resp.status_code == 409
    assert "not completed" in resp.json()["detail"]

    # A legacy evaluation has no prediction link to compare through.
    legacy = client.post(
        "/evaluations",
        json={"model_version_id": "m:v1", "dataset_uri": "ds", "metrics": ["accuracy"]},
    ).json()
    resp = client.get("/compare", params={"a": a, "b": legacy["id"]})
    assert resp.status_code == 422
    assert "no prediction link" in resp.json()["detail"]
