"""SDK unit tests that don't require a running control plane."""

from __future__ import annotations

import pytest

import localml as ml
from localml.config import Config, configure
from localml.exceptions import EvaluationFailedError, LocalMLError
from localml.types import EvaluationJob, ModelVersion


def test_public_api_surface():
    for name in ("start_run", "log_metrics", "evaluate", "deploy", "configure", "datasets"):
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


def test_config_precedence_file_fallback(monkeypatch):
    """With no explicit arg or env var, values come from ~/.localml/config.toml."""
    from localml import config as cfg_mod

    monkeypatch.delenv("LOCALML_API_URL", raising=False)
    monkeypatch.delenv("LOCALML_API_TOKEN", raising=False)
    monkeypatch.setattr(
        cfg_mod,
        "_load_file",
        lambda *a, **k: {
            "api_url": "http://file:1",
            "token": "ftok",
            "timeout": 12,
            "max_retries": 7,
        },
    )
    cfg = cfg_mod.configure()
    assert cfg.api_url == "http://file:1"
    assert cfg.token == "ftok"
    assert cfg.timeout == 12.0
    assert cfg.max_retries == 7


def test_config_precedence_env_over_file(monkeypatch):
    from localml import config as cfg_mod

    monkeypatch.setenv("LOCALML_API_URL", "http://env:2")
    monkeypatch.setattr(cfg_mod, "_load_file", lambda *a, **k: {"api_url": "http://file:1"})
    assert cfg_mod.configure().api_url == "http://env:2"


def test_config_precedence_explicit_over_all(monkeypatch):
    from localml import config as cfg_mod

    monkeypatch.setenv("LOCALML_API_URL", "http://env:2")
    monkeypatch.setattr(cfg_mod, "_load_file", lambda *a, **k: {"api_url": "http://file:1"})
    assert cfg_mod.configure(api_url="http://explicit:3").api_url == "http://explicit:3"


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


def test_evaluation_wait_timeout_raises():
    # A client that never leaves 'running' should trip the deadline.
    job = EvaluationJob(
        id="j3", model_version_id="mv", status="running", _client=_FakeClient("running")
    )
    with pytest.raises(EvaluationFailedError, match="timed out"):
        job.wait(timeout=0.05, poll_interval=0.01)


def test_model_version_is_dataholder():
    mv = ModelVersion(id="1", model_name="m", version=1, framework="mlx", artifact_uri="u")
    assert mv.status == "created"


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    from localml._hashing import sha256_file

    f = tmp_path / "blob.bin"
    payload = b"the quick brown fox" * 5000  # exceed the read chunk size
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()
