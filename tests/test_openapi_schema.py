"""The checked-in OpenAPI schema must match the live app.

``docs/openapi.json`` is a committed artifact (see ``scripts/export_openapi.py``). If a route or
schema changes without regenerating it, this test fails with the fix command, so the contract
can never silently drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_openapi", ROOT / "scripts" / "export_openapi.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_openapi_matches_app():
    exporter = _load_exporter()
    assert exporter.SCHEMA_PATH.exists(), (
        "run scripts/export_openapi.py to create docs/openapi.json"
    )
    current = exporter.SCHEMA_PATH.read_text()
    assert current == exporter.render_schema(), (
        "docs/openapi.json is out of date; run `uv run python scripts/export_openapi.py`"
    )
