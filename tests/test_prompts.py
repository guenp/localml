"""Prompt registry tests — sandboxed templating unit tests plus the API endpoints."""

from __future__ import annotations

import pytest
from app.templating import TemplateError, extract_variables, render

# -- templating ------------------------------------------------------------------


def test_extract_variables_in_first_use_order():
    tpl = "System: {persona}\nQ: {question}\nContext: {question} for {persona}"
    assert extract_variables(tpl) == ["persona", "question"]


def test_extract_variables_ignores_escaped_braces():
    assert extract_variables('Return JSON like {{"answer": ...}} for {question}') == ["question"]


def test_format_spec_is_allowed():
    assert render("score: {score:.2f}", {"score": 0.98765}) == "score: 0.99"


@pytest.mark.parametrize(
    "template",
    [
        "hello {}",  # positional
        "hello {0}",  # numbered
        "hello {user.name}",  # attribute access
        "hello {rows[0]}",  # index access
        "hello {x:{width}}",  # nested field in format spec
        "unbalanced {brace",  # malformed
    ],
)
def test_unsafe_or_malformed_templates_rejected(template):
    with pytest.raises(TemplateError):
        extract_variables(template)


def test_render_reports_missing_and_extra_variables():
    with pytest.raises(TemplateError, match="missing: question"):
        render("Q: {question}", {})
    with pytest.raises(TemplateError, match="extra: bogus"):
        render("Q: {question}", {"question": "hi", "bogus": 1})


# -- API endpoints ----------------------------------------------------------------


def _register(client, name="qa", template="Q: {question}\nA:", **kwargs):
    return client.post("/prompts", json={"name": name, "template": template, **kwargs})


def test_prompt_registration_extracts_variables_and_versions(client):
    v1 = _register(client).json()
    assert v1["version"] == "v1"
    assert v1["variables"] == ["question"]

    v2 = _register(client, template="Q: {question}\nContext: {context}\nA:").json()
    assert v2["version"] == "v2"
    assert v2["variables"] == ["question", "context"]

    versions = client.get("/prompts/qa").json()
    assert [p["version"] for p in versions] == ["v1", "v2"]


def test_prompt_explicit_duplicate_version_conflicts(client):
    assert _register(client, version="v1").status_code == 201
    assert _register(client, version="v1").status_code == 409


def test_prompt_rejects_unsafe_template(client):
    resp = _register(client, template="hello {user.name}")
    assert resp.status_code == 422
    assert "attribute or index access" in resp.json()["detail"]


def test_prompt_idempotency_key(client):
    headers = {"Idempotency-Key": "prompt-1"}
    r1 = client.post("/prompts", json={"name": "p", "template": "{x}"}, headers=headers).json()
    r2 = client.post("/prompts", json={"name": "p", "template": "{x}"}, headers=headers).json()
    assert r1["id"] == r2["id"]
    assert r2["version"] == "v1"


def test_prompt_get_unknown_404(client):
    assert client.get("/prompts/nope").status_code == 404


def test_prompt_version_lookup_and_render(client):
    _register(client, template="Hello {name}, question: {question}")
    got = client.get("/prompts/qa/versions/v1").json()
    assert got["variables"] == ["name", "question"]

    rendered = client.post(
        "/prompts/qa/versions/v1/render",
        json={"variables": {"name": "Ada", "question": "why?"}},
    ).json()
    assert rendered == {"name": "qa", "version": "v1", "rendered": "Hello Ada, question: why?"}


def test_prompt_render_rejects_variable_mismatch(client):
    _register(client, template="Q: {question}")
    missing = client.post("/prompts/qa/versions/v1/render", json={"variables": {}})
    assert missing.status_code == 422
    assert "missing: question" in missing.json()["detail"]

    extra = client.post(
        "/prompts/qa/versions/v1/render",
        json={"variables": {"question": "hi", "bogus": 1}},
    )
    assert extra.status_code == 422
    assert "extra: bogus" in extra.json()["detail"]


def test_prompt_resolution(client):
    created = _register(client).json()
    resolved = client.post(
        "/resolve", json={"resource_type": "prompt", "reference": "qa:v1"}
    ).json()
    assert resolved["id"] == created["id"]
    assert resolved["version"] == "v1"
