"""Dashboard data-helper + CLI-launcher tests.

The Streamlit UI itself (``dashboard.main``) is only exercised under ``streamlit run`` and is not
unit-tested, but the pure client helpers and the ``localml dashboard`` launcher are.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from localml import cli, dashboard

runner = CliRunner()


class _StubClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path, kwargs))
        if self.fail:
            raise RuntimeError("boom")
        return {"ok": True, "path": path}


@pytest.fixture
def stub(monkeypatch):  # type: ignore[no-untyped-def]
    client = _StubClient()
    monkeypatch.setattr(dashboard, "get_client", lambda: client)
    return client


def test_get_helper_passes_params(stub):
    assert dashboard._get("/compare", params={"a": "x"}) == {"ok": True, "path": "/compare"}
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("GET", "/compare")
    assert kwargs["params"] == {"a": "x"}


def test_post_helper(stub):
    dashboard._post("/deployments/d/predict", {"prompt": "hi"})
    method, path, kwargs = stub.requests[0]
    assert (method, path) == ("POST", "/deployments/d/predict")
    assert kwargs["json"] == {"prompt": "hi"}


def test_helpers_surface_errors_as_dict(monkeypatch):
    monkeypatch.setattr(dashboard, "get_client", lambda: _StubClient(fail=True))
    assert dashboard._get("/health") == {"error": "boom"}
    assert dashboard._post("/x", {}) == {"error": "boom"}


def test_dashboard_launch_invokes_streamlit(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/streamlit")
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    result = runner.invoke(cli.app, ["dashboard", "--port", "9000"])
    assert result.exit_code == 0
    assert calls[0][:2] == ["streamlit", "run"]
    assert calls[0][-2:] == ["--server.port", "9000"]
    assert calls[0][2].endswith("dashboard.py")


def test_dashboard_errors_without_streamlit(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    ran = False

    def _fail(*a, **k):
        nonlocal ran
        ran = True

    monkeypatch.setattr(cli.subprocess, "run", _fail)
    result = runner.invoke(cli.app, ["dashboard"])
    assert result.exit_code != 0
    assert not ran


def test_dashboard_module_import_does_not_require_streamlit():
    # The module imported at the top of this file with no `streamlit` installed, proving the
    # extra is only needed under `streamlit run`. `main` defers the import to call time.
    import sys

    assert "streamlit" not in sys.modules
    assert callable(dashboard.main)
