"""Serving proxy, backend registry, deployment flow, and hot swap (Phase 4)."""

from __future__ import annotations

import json

import httpx
import pytest
from app import serving
from app.db import Deployment, Model, ModelVersion, Project
from app.serving import register_backend, resolve_backend

# -- backend registry ---------------------------------------------------------------


def _deployment(target: str = "local", config: dict | None = None) -> Deployment:
    project = Project(name="local")
    model = Model(project_id="p", name="assistant")
    project.id = "p"
    mv = ModelVersion(
        model_id="m", version=1, framework="mlx", artifact_uri="file:///m", status="staging"
    )
    mv.model = model
    dep = Deployment(model_version_id="mv", target=target, status="active", config=config or {})
    dep.id = "dep-1"
    dep.model_version = mv
    return dep


def test_resolve_backend_precedence(monkeypatch):
    monkeypatch.setattr(serving, "_BACKENDS", {})  # isolate from other tests' registrations
    # Nothing registered -> falls back to the serving URL, model defaults to the model name.
    backend = resolve_backend(_deployment())
    assert backend.base_url == serving.settings.serving_url
    assert backend.model == "assistant"

    # A registered target backend is used and can set the model.
    register_backend("prod", base_url="http://gpu:8000", model="llama3")
    backend = resolve_backend(_deployment(target="prod"))
    assert (backend.base_url, backend.model) == ("http://gpu:8000", "llama3")

    # The deployment's own config overrides the registered backend.
    backend = resolve_backend(
        _deployment(target="prod", config={"base_url": "http://local:11434", "model": "mistral"})
    )
    assert (backend.base_url, backend.model) == ("http://local:11434", "mistral")


def test_check_backend_health(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(handler))
    assert serving.check_backend_health(_deployment()) is True

    # A 404 (backend up, no /v1/models) still counts as reachable; a 5xx does not.
    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(lambda r: httpx.Response(404)))
    assert serving.check_backend_health(_deployment()) is True
    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(lambda r: httpx.Response(500)))
    assert serving.check_backend_health(_deployment()) is False


def test_check_backend_health_unreachable(monkeypatch):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(boom))
    assert serving.check_backend_health(_deployment()) is False


# -- deployment flow + proxy (through the API) --------------------------------------


@pytest.fixture
def staged_model(client):
    """A model version promoted to a deployable (staging) state; returns its id."""
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///m"}
    mv = client.post("/models/m/versions", json=body).json()
    client.post("/models/m/versions/1/promote", json={"target_status": "candidate"})
    client.post("/models/m/versions/1/promote", json={"target_status": "staging"})
    return mv["id"]


def test_create_deployment_health_check_sets_status(client, staged_model, monkeypatch):
    from app.routers import deployments

    monkeypatch.setattr(deployments, "check_backend_health", lambda dep: False)
    resp = client.post("/deployments", json={"model_version_id": staged_model, "target": "local"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "degraded"  # backend did not answer
    assert body["endpoint_url"] == f"/deployments/{body['id']}/v1/chat/completions"


def test_chat_completions_proxies_to_backend(client, staged_model, monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(handler))
    dep = client.post(
        "/deployments",
        json={"model_version_id": staged_model, "target": "local", "config": {"model": "llama3"}},
    ).json()

    resp = client.post(
        f"/deployments/{dep['id']}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hey"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi"
    assert captured["path"] == "/v1/chat/completions"
    # The backend model id is injected when the caller didn't set one.
    assert captured["body"]["model"] == "llama3"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hey"}]


def test_predict_sugar_wraps_prompt_in_messages(client, staged_model, monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(handler))
    dep = client.post(
        "/deployments", json={"model_version_id": staged_model, "target": "local"}
    ).json()

    resp = client.post(f"/deployments/{dep['id']}/predict", json={"prompt": "2+2?"})
    assert resp.status_code == 200
    assert captured["body"]["messages"] == [{"role": "user", "content": "2+2?"}]
    assert "stream" not in captured["body"]


def test_predict_requires_prompt_or_messages(client, staged_model):
    dep = client.post(
        "/deployments", json={"model_version_id": staged_model, "target": "local"}
    ).json()
    resp = client.post(f"/deployments/{dep['id']}/predict", json={"temperature": 0.5})
    assert resp.status_code == 422
    assert "messages" in resp.json()["detail"]


def test_proxy_backend_unreachable_is_502(client, staged_model, monkeypatch):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(serving, "_transport", httpx.MockTransport(boom))
    dep = client.post(
        "/deployments", json={"model_version_id": staged_model, "target": "local"}
    ).json()
    resp = client.post(f"/deployments/{dep['id']}/predict", json={"prompt": "x"})
    assert resp.status_code == 502
    assert "unreachable" in resp.json()["detail"]


def test_proxy_on_inactive_deployment_is_409(client, staged_model):
    dep = client.post(
        "/deployments", json={"model_version_id": staged_model, "target": "local"}
    ).json()
    assert client.delete(f"/deployments/{dep['id']}").json()["status"] == "inactive"
    resp = client.post(f"/deployments/{dep['id']}/predict", json={"prompt": "x"})
    assert resp.status_code == 409


def test_proxy_unknown_deployment_is_404(client):
    assert client.post("/deployments/ghost/predict", json={"prompt": "x"}).status_code == 404


# -- hot swap -----------------------------------------------------------------------


def test_swap_repoints_model_and_config(client, monkeypatch):
    from app.routers import deployments

    # Two deployable versions of the same model.
    client.post(
        "/models/m/versions", json={"model_name": "m", "framework": "mlx", "artifact_uri": "u1"}
    )
    client.post(
        "/models/m/versions", json={"model_name": "m", "framework": "mlx", "artifact_uri": "u2"}
    )
    for v in (1, 2):
        client.post(f"/models/m/versions/{v}/promote", json={"target_status": "candidate"})
        client.post(f"/models/m/versions/{v}/promote", json={"target_status": "staging"})
    v1 = client.get("/models/m").json()["versions"][0]["id"]
    v2 = client.get("/models/m").json()["versions"][1]["id"]

    dep = client.post(
        "/deployments", json={"model_version_id": v1, "target": "local", "config": {"model": "a"}}
    ).json()

    monkeypatch.setattr(deployments, "check_backend_health", lambda d: True)
    swapped = client.patch(
        f"/deployments/{dep['id']}",
        json={"model_version_id": v2, "config": {"model": "b"}},
    ).json()
    assert swapped["model_version_id"] == v2
    assert swapped["config"]["model"] == "b"
    assert swapped["status"] == "active"

    # A partial patch merges config and leaves the model version untouched.
    swapped2 = client.patch(
        f"/deployments/{dep['id']}", json={"config": {"api_key": "secret"}}
    ).json()
    assert swapped2["model_version_id"] == v2
    assert swapped2["config"] == {"model": "b", "api_key": "secret"}


def test_swap_rejects_non_deployable_target(client, staged_model):
    # A fresh (created) version is not deployable.
    fresh = client.post(
        "/models/m/versions", json={"model_name": "m", "framework": "mlx", "artifact_uri": "u3"}
    ).json()["id"]
    dep = client.post(
        "/deployments", json={"model_version_id": staged_model, "target": "local"}
    ).json()
    resp = client.patch(f"/deployments/{dep['id']}", json={"model_version_id": fresh})
    assert resp.status_code == 409


def test_swap_unknown_deployment_is_404(client):
    assert client.patch("/deployments/ghost", json={"target": "x"}).status_code == 404
