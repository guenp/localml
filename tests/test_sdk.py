"""SDK unit tests that don't require a running control plane."""

from __future__ import annotations

import pytest

import localml as ml
from localml.config import Config, configure
from localml.exceptions import EvaluationFailedError, LocalMLError
from localml.types import EvaluationJob, ModelVersion


def test_public_api_surface():
    for name in ("start_run", "log_metrics", "evaluate", "deploy", "configure"):
        assert hasattr(ml, name)
    for fw in ("torch", "jax", "mlx", "huggingface"):
        assert hasattr(ml, fw)


def test_exception_hierarchy():
    assert issubclass(EvaluationFailedError, LocalMLError)


def test_configure_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALML_API_URL", raising=False)
    cfg = configure(api_url="http://example:9000", token="t")
    assert isinstance(cfg, Config)
    assert cfg.api_url == "http://example:9000"
    assert cfg.token == "t"


def test_configure_env_override(monkeypatch):
    monkeypatch.setenv("LOCALML_API_URL", "http://env:1234")
    cfg = configure()
    assert cfg.api_url == "http://env:1234"


class _FakeClient:
    """Returns a completed job on refresh."""

    def __init__(self, final_status="completed"):
        self.final_status = final_status

    def get_evaluation(self, job_id):
        return EvaluationJob(
            id=job_id, model_version_id="mv", status=self.final_status, metrics={"accuracy": 1.0}
        )


def test_evaluation_wait_success():
    job = EvaluationJob(
        id="j1", model_version_id="mv", status="running", _client=_FakeClient("completed")
    )
    result = job.wait(timeout=5, poll_interval=0.01)
    assert result.status == "completed"
    assert result.metrics == {"accuracy": 1.0}


def test_evaluation_wait_failure_raises():
    job = EvaluationJob(
        id="j2", model_version_id="mv", status="running", _client=_FakeClient("failed")
    )
    with pytest.raises(EvaluationFailedError):
        job.wait(timeout=5, poll_interval=0.01)


def test_model_version_is_dataholder():
    mv = ModelVersion(id="1", model_name="m", version=1, framework="mlx", artifact_uri="u")
    assert mv.status == "created"
