"""SDK↔API contract: every endpoint the SDK calls must exist in the OpenAPI schema.

This pins the SDK's route/method surface to the control plane's generated schema, so a rename
or signature change on either side fails fast here (complementing the live-server integration
tests, which additionally exercise payload shapes and responses).
"""

from __future__ import annotations

# (HTTP method, path template) pairs the localml client issues — see src/localml/client.py.
SDK_ENDPOINTS = [
    ("post", "/runs"),
    ("get", "/runs/{run_id}"),
    ("post", "/runs/{run_id}/metrics"),
    ("post", "/runs/{run_id}/params"),
    ("post", "/runs/{run_id}/artifacts"),
    ("post", "/models/{model_name}/versions"),
    ("get", "/models/{model_name}"),
    ("post", "/datasets"),
    ("get", "/datasets/{name}"),
    ("post", "/prompts"),
    ("get", "/prompts/{name}"),
    ("post", "/prompts/{name}/versions/{version}/render"),
    ("post", "/evaluations"),
    ("get", "/evaluations/{job_id}"),
    ("post", "/deployments"),
    ("post", "/deployments/{deployment_id}/predict"),
]


def test_sdk_endpoints_exist_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    missing = [
        (method, path)
        for method, path in SDK_ENDPOINTS
        if path not in paths or method not in paths[path]
    ]
    assert not missing, f"SDK endpoints absent from the OpenAPI schema: {missing}"
