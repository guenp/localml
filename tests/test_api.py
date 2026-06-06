"""API control-plane tests exercising the full demo flow."""

from __future__ import annotations


def test_health(client):  # type: ignore[no-untyped-def]
    assert client.get("/health").json() == {"status": "ok"}


def test_run_lifecycle(client):  # type: ignore[no-untyped-def]
    run = client.post("/runs", json={"project": "demo", "config": {"lr": 0.1}}).json()
    rid = run["id"]
    assert run["status"] == "running"

    assert client.post(f"/runs/{rid}/metrics", json={"metrics": {"acc": 0.9}}).status_code == 200
    assert client.post(f"/runs/{rid}/params", json={"params": {"bs": 4}}).status_code == 200

    fetched = client.get(f"/runs/{rid}").json()
    assert fetched["project"] == "demo"


def test_model_registration_and_versioning(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    v1 = client.post("/models/m/versions", json=body).json()
    v2 = client.post("/models/m/versions", json=body).json()
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert v1["status"] == "created"

    model = client.get("/models/m").json()
    assert len(model["versions"]) == 2


def test_promotion_rules(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    client.post("/models/m/versions", json=body)

    # created -> production is invalid (422)
    bad = client.post("/models/m/versions/1/promote", json={"target_status": "production"})
    assert bad.status_code == 422

    # created -> candidate -> staging is valid
    promote = "/models/m/versions/1/promote"
    assert client.post(promote, json={"target_status": "candidate"}).status_code == 200
    assert client.post(promote, json={"target_status": "staging"}).status_code == 200


def test_deployment_requires_deployable_state(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    mv = client.post("/models/m/versions", json=body).json()

    # created is not deployable -> 409
    conflict = client.post("/deployments", json={"model_version_id": mv["id"], "target": "local"})
    assert conflict.status_code == 409

    client.post("/models/m/versions/1/promote", json={"target_status": "candidate"})
    client.post("/models/m/versions/1/promote", json={"target_status": "staging"})

    dep = client.post("/deployments", json={"model_version_id": mv["id"], "target": "local"})
    assert dep.status_code == 201
    assert dep.json()["status"] == "active"


def test_evaluation_job_created(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    mv = client.post("/models/m/versions", json=body).json()
    job = client.post(
        "/evaluations",
        json={"model_version_id": mv["id"], "dataset_uri": "ds", "metrics": ["accuracy"]},
    ).json()
    assert job["status"] == "queued"
    assert client.get(f"/evaluations/{job['id']}").json()["model_version_id"] == mv["id"]


def test_evaluation_unknown_model_version(client):  # type: ignore[no-untyped-def]
    resp = client.post(
        "/evaluations",
        json={"model_version_id": "nope", "dataset_uri": "ds", "metrics": ["accuracy"]},
    )
    assert resp.status_code == 404
