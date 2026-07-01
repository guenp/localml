"""API control-plane tests exercising the full local flow."""

from __future__ import annotations


def test_health(client):  # type: ignore[no-untyped-def]
    assert client.get("/health").json() == {"status": "ok"}


def test_run_lifecycle(client):  # type: ignore[no-untyped-def]
    run = client.post("/runs", json={"project": "local", "config": {"lr": 0.1}}).json()
    rid = run["id"]
    assert run["status"] == "running"

    assert client.post(f"/runs/{rid}/metrics", json={"metrics": {"acc": 0.9}}).status_code == 200
    assert client.post(f"/runs/{rid}/params", json={"params": {"bs": 4}}).status_code == 200

    fetched = client.get(f"/runs/{rid}").json()
    assert fetched["project"] == "local"


def test_model_registration_and_versioning(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    v1 = client.post("/models/m/versions", json=body).json()
    v2 = client.post("/models/m/versions", json=body).json()
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert v1["status"] == "created"

    model = client.get("/models/m").json()
    assert len(model["versions"]) == 2


def test_idempotent_model_registration(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    headers = {"Idempotency-Key": "same-create"}
    v1 = client.post("/models/m/versions", json=body, headers=headers).json()
    v2 = client.post("/models/m/versions", json=body, headers=headers).json()
    assert v1["id"] == v2["id"]
    assert v2["version"] == 1


def test_dataset_registration_and_resolution(client):  # type: ignore[no-untyped-def]
    dataset = client.post(
        "/datasets",
        json={
            "project": "local",
            "name": "eval-set",
            "artifact_uri": "s3://localml-artifacts/datasets/eval-set.jsonl",
            "rows": [{"prompt": "a", "expected": "b"}, {"example_id": "known", "prompt": "c"}],
        },
    ).json()
    assert dataset["version"] == "v1"
    assert dataset["row_count"] == 2
    assert dataset["example_ids"][1] == "known"

    resolved = client.post(
        "/resolve", json={"resource_type": "dataset", "reference": "eval-set:v1"}
    ).json()
    assert resolved["id"] == dataset["id"]


def test_model_name_version_resolution(client):  # type: ignore[no-untyped-def]
    body = {"model_name": "m", "framework": "mlx", "artifact_uri": "file:///tmp/m"}
    mv = client.post("/models/m/versions", json=body).json()
    resolved = client.post("/resolve", json={"resource_type": "model", "reference": "m:v1"}).json()
    assert resolved["id"] == mv["id"]


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


def test_run_status_is_durable(client):  # type: ignore[no-untyped-def]
    """A status change written on one request is visible on a later GET (DB-backed)."""
    rid = client.post("/runs", json={"project": "local", "config": {}}).json()["id"]
    client.post(f"/runs/{rid}/metrics", json={"metrics": {"acc": 0.9}, "status": "completed"})
    assert client.get(f"/runs/{rid}").json()["status"] == "completed"


def test_artifact_registration_returns_record(client):  # type: ignore[no-untyped-def]
    rid = client.post("/runs", json={"project": "local", "config": {}}).json()["id"]
    resp = client.post(
        f"/runs/{rid}/artifacts",
        json={"uri": "file:///tmp/model.safetensors", "artifact_type": "model"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] and body["uri"].endswith("model.safetensors")
    # No MinIO reachable in unit tests -> presign degrades to None.
    assert body["upload_url"] is None


def test_dataset_conflict_on_duplicate_version(client):  # type: ignore[no-untyped-def]
    body = {"project": "local", "name": "ds", "artifact_uri": "u", "version": "v1", "rows": []}
    assert client.post("/datasets", json=body).status_code == 201
    assert client.post("/datasets", json=body).status_code == 409


def test_dataset_idempotency_key(client):  # type: ignore[no-untyped-def]
    body = {"project": "local", "name": "ds", "artifact_uri": "u", "rows": [{"p": "a"}]}
    headers = {"Idempotency-Key": "ds-create"}
    first = client.post("/datasets", json=body, headers=headers).json()
    second = client.post("/datasets", json=body, headers=headers).json()
    assert first["id"] == second["id"]
    # A second dataset without the key still auto-increments the version.
    third = client.post("/datasets", json=body).json()
    assert third["version"] == "v2"


def test_idempotency_conflict_on_body_mismatch(client):  # type: ignore[no-untyped-def]
    headers = {"Idempotency-Key": "reused"}
    client.post("/projects", json={"name": "a"}, headers=headers)
    clash = client.post("/projects", json={"name": "b"}, headers=headers)
    assert clash.status_code == 409


def test_evaluation_and_deployment_idempotency(client):  # type: ignore[no-untyped-def]
    mv = client.post(
        "/models/m/versions",
        json={"model_name": "m", "framework": "mlx", "artifact_uri": "u"},
    ).json()
    eval_body = {"model_version_id": mv["id"], "dataset_uri": "ds", "metrics": ["accuracy"]}
    ehead = {"Idempotency-Key": "eval-1"}
    e1 = client.post("/evaluations", json=eval_body, headers=ehead).json()
    e2 = client.post("/evaluations", json=eval_body, headers=ehead).json()
    assert e1["id"] == e2["id"]

    client.post("/models/m/versions/1/promote", json={"target_status": "candidate"})
    client.post("/models/m/versions/1/promote", json={"target_status": "staging"})
    dep_body = {"model_version_id": mv["id"], "target": "local"}
    dhead = {"Idempotency-Key": "dep-1"}
    d1 = client.post("/deployments", json=dep_body, headers=dhead).json()
    d2 = client.post("/deployments", json=dep_body, headers=dhead).json()
    assert d1["id"] == d2["id"]


def test_models_are_project_scoped(client):  # type: ignore[no-untyped-def]
    """Registering with an explicit project still resolves by name:version."""
    client.post(
        "/models/scoped/versions",
        json={"model_name": "scoped", "framework": "mlx", "artifact_uri": "u", "project": "proj-a"},
    )
    resolved = client.post(
        "/resolve", json={"resource_type": "model", "reference": "scoped:v1"}
    ).json()
    assert resolved["name"] == "scoped"
    assert resolved["version"] == "v1"
