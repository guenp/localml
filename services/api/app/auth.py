"""Minimal bearer-token auth for the MVP.

When ``LOCALML_AUTH_BYPASS=true`` (the default for local demos) all requests are allowed.
Otherwise the ``Authorization: Bearer <token>`` header must match ``LOCALML_API_TOKEN``.
Future work (OIDC, mTLS, RBAC) is tracked in the roadmap.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import settings


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if settings.auth_bypass:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.api_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
