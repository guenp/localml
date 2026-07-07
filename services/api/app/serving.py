"""Local serving: backend registry + OpenAI-compatible proxy (Phase 4).

Serving is a thin proxy, not a bespoke inference server. The control plane resolves a
deployment to a backend ``{base_url, model, api_key}`` and forwards ``/v1/chat/completions``
and ``/v1/completions`` bodies unchanged — streaming included. Ollama, MLX-LM, llama.cpp,
and vLLM all speak this API, so one proxy covers every local backend.

Backends are resolved **at request time** with this precedence: the deployment's own
``config`` overrides → the registered target backend → the global serving URL, with the
model id defaulting to the deployed model's registry name. Hot model swap is therefore just
a PATCH that repoints the deployment's backend/model; no process restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException, status
from fastapi.responses import Response

from .config import settings
from .db import Deployment

log = logging.getLogger("localml.serving")

DEFAULT_TIMEOUT = 120.0
HEALTH_TIMEOUT = 5.0

# Injectable transport for tests (httpx.MockTransport serves both sync and async clients);
# None = real network.
_transport: Any | None = None


@dataclass
class Backend:
    """An OpenAI-compatible serving runtime."""

    base_url: str
    model: str | None = None  # None = default to the deployed model's registry name
    api_key: str | None = None


_BACKENDS: dict[str, Backend] = {}


def register_backend(
    target: str, *, base_url: str, model: str | None = None, api_key: str | None = None
) -> None:
    """Map a deployment target to a serving backend."""
    _BACKENDS[target] = Backend(base_url=base_url, model=model, api_key=api_key)


def resolve_backend(deployment: Deployment) -> Backend:
    """Resolve a deployment to its backend (see module docstring for the precedence)."""
    backend = _BACKENDS.get(deployment.target, Backend(base_url=settings.serving_url))
    config = deployment.config or {}
    return Backend(
        base_url=config.get("base_url") or backend.base_url,
        model=config.get("model") or backend.model or deployment.model_version.model.name,
        api_key=config.get("api_key") or backend.api_key,
    )


def check_backend_health(deployment: Deployment) -> bool:
    """``GET {base_url}/v1/models`` with a short timeout; ``False`` when unreachable."""
    backend = resolve_backend(deployment)
    try:
        with httpx.Client(
            base_url=backend.base_url, timeout=HEALTH_TIMEOUT, transport=_transport
        ) as client:
            return client.get("/v1/models").status_code < 500
    except Exception as exc:
        log.warning("serving backend %s health check failed: %s", backend.base_url, exc)
        return False


async def proxy_openai(deployment: Deployment, path: str, payload: dict[str, Any]) -> Response:
    """Forward an OpenAI-style request body to the deployment's backend and return its reply.

    The backend's model id is injected when the caller didn't set one, so clients can talk to
    a deployment without knowing which model it currently points at. The upstream response is
    buffered and returned verbatim (status, content type, body) — adequate at single-
    workstation scale; SSE passthrough streaming is a future enhancement.
    """
    backend = resolve_backend(deployment)
    body = dict(payload)
    body.setdefault("model", backend.model)
    headers = {"Authorization": f"Bearer {backend.api_key}"} if backend.api_key else {}

    async with httpx.AsyncClient(
        base_url=backend.base_url, timeout=DEFAULT_TIMEOUT, transport=_transport
    ) as client:
        try:
            upstream = await client.post(path, json=body, headers=headers)
        except httpx.TransportError as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"serving backend unreachable at {backend.base_url}: {exc}",
            ) from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
