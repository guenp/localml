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


def test_evals_register_metric_validates(monkeypatch):
    monkeypatch.setattr(ml.evals, "_LOCAL_METRICS", {})
    ml.evals.register_metric("mine", lambda records, config: 1.0)
    assert "mine" in ml.evals._LOCAL_METRICS
    with pytest.raises(TypeError):
        ml.evals.register_metric("bad", "not callable")  # type: ignore[arg-type]


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


# -- upload_file ------------------------------------------------------------------


def _upload_client(max_retries: int = 3):  # type: ignore[no-untyped-def]
    from localml.client import Client

    return Client(Config(api_url="http://unused", token=None, timeout=1, max_retries=max_retries))


def test_upload_file_streams_from_disk(tmp_path, monkeypatch):
    import httpx

    from localml import client as client_mod

    f = tmp_path / "bundle.tar.gz"
    f.write_bytes(b"payload-bytes")
    seen = {}

    def fake_put(url, content=None, timeout=None):  # type: ignore[no-untyped-def]
        assert hasattr(content, "read")  # a file object, not the whole file in memory
        seen["body"] = content.read()
        return httpx.Response(200, request=httpx.Request("PUT", url))

    monkeypatch.setattr(client_mod.httpx, "put", fake_put)
    _upload_client().upload_file("http://minio/presigned", str(f))
    assert seen["body"] == b"payload-bytes"


def test_upload_file_retries_transient_then_succeeds(tmp_path, monkeypatch):
    import httpx

    from localml import client as client_mod

    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    statuses = iter([503, 200])
    attempts = []

    def fake_put(url, content=None, timeout=None):  # type: ignore[no-untyped-def]
        attempts.append(content.read())
        return httpx.Response(next(statuses), request=httpx.Request("PUT", url))

    monkeypatch.setattr(client_mod.httpx, "put", fake_put)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    _upload_client().upload_file("http://minio/presigned", str(f))
    # The file is reopened per attempt, so the retry re-sends the full payload.
    assert attempts == [b"x", b"x"]


def test_upload_file_client_error_fails_without_retry(tmp_path, monkeypatch):
    import httpx

    from localml import client as client_mod
    from localml.exceptions import ArtifactUploadError

    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    calls = []

    def fake_put(url, content=None, timeout=None):  # type: ignore[no-untyped-def]
        calls.append(1)
        return httpx.Response(403, text="denied", request=httpx.Request("PUT", url))

    monkeypatch.setattr(client_mod.httpx, "put", fake_put)
    with pytest.raises(ArtifactUploadError, match="403"):
        _upload_client().upload_file("http://minio/presigned", str(f))
    assert calls == [1]  # 403 is not transient; no pointless retries
