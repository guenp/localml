"""CLI tests.

These stub the client transport, so they verify command wiring — argument parsing, request
method/path/payload — without a server.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from localml import cli

runner = CliRunner()


class _StubClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path, kwargs))
        return {"ok": True}


@pytest.fixture
def stub(monkeypatch):  # type: ignore[no-untyped-def]
    client = _StubClient()
    monkeypatch.setattr(cli, "get_client", lambda: client)
    return client


def test_prompts_register_inline(stub):
    result = runner.invoke(cli.app, ["prompts", "register", "qa", "--template", "Q: {question}"])
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/prompts")
    assert kwargs["json"]["template"] == "Q: {question}"
    assert kwargs["json"]["project"] == "local"
    assert kwargs["idempotent"] is True


def test_prompts_register_from_file(stub, tmp_path):
    tpl = tmp_path / "prompt.txt"
    tpl.write_text("Hello {name}")
    result = runner.invoke(cli.app, ["prompts", "register", "greet", "--file", str(tpl)])
    assert result.exit_code == 0
    assert stub.requests[0][2]["json"]["template"] == "Hello {name}"


def test_prompts_register_requires_exactly_one_source(stub, tmp_path):
    neither = runner.invoke(cli.app, ["prompts", "register", "qa"])
    assert neither.exit_code != 0

    tpl = tmp_path / "p.txt"
    tpl.write_text("x")
    both = runner.invoke(
        cli.app, ["prompts", "register", "qa", "--template", "x", "--file", str(tpl)]
    )
    assert both.exit_code != 0
    assert not stub.requests


def test_prompts_get(stub):
    result = runner.invoke(cli.app, ["prompts", "get", "qa"])
    assert result.exit_code == 0
    assert stub.requests[0][:2] == ("GET", "/prompts/qa")


def test_prompts_render_parses_vars(stub):
    result = runner.invoke(
        cli.app,
        ["prompts", "render", "qa", "v1", "--var", "question=why?", "--var", "note=a=b"],
    )
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/prompts/qa/versions/v1/render")
    # Values may themselves contain '='; only the first one splits.
    assert kwargs["json"]["variables"] == {"question": "why?", "note": "a=b"}


def test_prompts_render_rejects_malformed_var(stub):
    result = runner.invoke(cli.app, ["prompts", "render", "qa", "v1", "--var", "no-equals"])
    assert result.exit_code != 0
    assert not stub.requests


def test_predictions_run(stub):
    result = runner.invoke(
        cli.app,
        [
            "predictions",
            "run",
            "m:v1",
            "evalset:v1",
            "qa:v2",
            "--provider",
            "echo",
            "--config",
            '{"batch_size": 8}',
        ],
    )
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/predictions")
    assert kwargs["json"] == {
        "model": "m:v1",
        "dataset": "evalset:v1",
        "prompt": "qa:v2",
        "provider": "echo",
        "config": {"batch_size": 8},
    }
    assert kwargs["idempotent"] is True


def test_predictions_run_rejects_malformed_config(stub):
    result = runner.invoke(
        cli.app, ["predictions", "run", "m:v1", "d:v1", "p:v1", "--config", "not-json"]
    )
    assert result.exit_code != 0
    assert not stub.requests


def test_predictions_status_and_results(stub):
    assert runner.invoke(cli.app, ["predictions", "status", "j1"]).exit_code == 0
    assert runner.invoke(cli.app, ["predictions", "results", "j1"]).exit_code == 0
    assert [r[:2] for r in stub.requests] == [
        ("GET", "/predictions/j1"),
        ("GET", "/predictions/j1/results"),
    ]


def test_evals_run(stub):
    result = runner.invoke(
        cli.app,
        [
            "evals",
            "run",
            "job-1",
            "-m",
            "exact_match",
            "--metric",
            "error_rate",
            "--config",
            '{"expected_field": "label"}',
        ],
    )
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/evaluations")
    assert kwargs["json"] == {
        "prediction": "job-1",
        "metrics": ["exact_match", "error_rate"],
        "config": {"expected_field": "label"},
    }
    assert kwargs["idempotent"] is True


def test_evals_run_rejects_malformed_config(stub):
    result = runner.invoke(
        cli.app, ["evals", "run", "job-1", "-m", "error_rate", "--config", "not-json"]
    )
    assert result.exit_code != 0
    assert not stub.requests


def test_evals_status(stub):
    assert runner.invoke(cli.app, ["evals", "status", "e1"]).exit_code == 0
    assert stub.requests[0][:2] == ("GET", "/evaluations/e1")


def test_deployments_create(stub):
    result = runner.invoke(
        cli.app,
        ["deployments", "create", "m:v1", "--target", "local", "--config", '{"model": "llama3"}'],
    )
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/deployments")
    assert kwargs["json"] == {
        "model_version_id": "m:v1",
        "target": "local",
        "config": {"model": "llama3"},
    }
    assert kwargs["idempotent"] is True


def test_deployments_swap(stub):
    result = runner.invoke(
        cli.app, ["deployments", "swap", "dep-1", "--model", "m:v2", "--config", '{"model": "b"}']
    )
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("PATCH", "/deployments/dep-1")
    assert kwargs["json"] == {"model_version_id": "m:v2", "target": None, "config": {"model": "b"}}


def test_deployments_predict(stub):
    result = runner.invoke(cli.app, ["deployments", "predict", "dep-1", "2+2?"])
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/deployments/dep-1/predict")
    assert kwargs["json"] == {"prompt": "2+2?"}


def test_compare_command(stub):
    result = runner.invoke(cli.app, ["compare", "a1", "b1", "--max-examples", "5"])
    assert result.exit_code == 0
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("GET", "/compare")
    assert kwargs["params"] == {"a": "a1", "b": "b1", "max_examples": 5}


def test_runs_and_models_get(stub):
    assert runner.invoke(cli.app, ["runs", "get", "r1"]).exit_code == 0
    assert runner.invoke(cli.app, ["models", "get", "m"]).exit_code == 0
    assert [r[:2] for r in stub.requests] == [("GET", "/runs/r1"), ("GET", "/models/m")]
