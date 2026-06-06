"""Seed a default local user and demo project.

Scaffold version targets the running control plane over HTTP so it works against the
in-memory store. Phase 1 will seed Postgres directly for reproducible demos.
"""

from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("LOCALML_API_URL", "http://localhost:8000")
TOKEN = os.environ.get("LOCALML_API_TOKEN", "local-dev-token")


def main() -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(base_url=API_URL, headers=headers, timeout=10) as client:
        resp = client.post("/projects", json={"name": "local-demo", "description": "Demo project"})
        resp.raise_for_status()
        print("seeded project:", resp.json())


if __name__ == "__main__":
    main()
