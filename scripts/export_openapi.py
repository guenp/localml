"""Export the control plane's OpenAPI schema to a checked-in file.

The generated schema (``docs/openapi.json``) is committed so the SDK's route/method surface,
API docs, and any external client generators have a stable, reviewable contract. A test
(``tests/test_openapi_schema.py``) regenerates the schema and fails if the checked-in copy has
drifted, so this script is the one place to re-run after changing a route:

    uv run python scripts/export_openapi.py

Run without arguments to write ``docs/openapi.json``; pass ``--check`` to verify it is current
without writing (exit 1 on drift).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "docs" / "openapi.json"
sys.path.insert(0, str(ROOT / "services" / "api"))


def render_schema() -> str:
    """Return the app's OpenAPI schema as pretty-printed, newline-terminated JSON."""
    from app.main import app

    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    rendered = render_schema()
    if "--check" in argv:
        current = SCHEMA_PATH.read_text() if SCHEMA_PATH.exists() else ""
        if current != rendered:
            print(
                f"{SCHEMA_PATH.relative_to(ROOT)} is out of date; "
                "run `uv run python scripts/export_openapi.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{SCHEMA_PATH.relative_to(ROOT)} is up to date")
        return 0
    SCHEMA_PATH.write_text(rendered)
    print(f"wrote {SCHEMA_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
