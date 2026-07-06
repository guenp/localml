"""Inference providers for prediction jobs (and, in Phase 4, the serving proxy).

One interface — :class:`InferenceProvider` — with two built-ins:

- ``openai`` (default): a thin client for any **OpenAI-compatible** backend
  (``POST {base_url}/v1/chat/completions``). Ollama, MLX-LM, llama.cpp, and vLLM all speak
  this API, so a single provider covers every local runtime with no bespoke protocol.
- ``echo``: deterministic, dependency-free — returns the rendered prompt as the output.
  Used for pipeline smoke tests and CI, where no model server is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import settings


class InferenceError(RuntimeError):
    """A provider failed to produce an output for one example."""


@dataclass
class InferenceResult:
    output: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class InferenceProvider(Protocol):
    def generate(self, prompt: str, config: dict[str, Any]) -> InferenceResult: ...


class EchoProvider:
    """Deterministic stand-in: the output is the rendered prompt itself."""

    def generate(self, prompt: str, config: dict[str, Any]) -> InferenceResult:
        start = time.perf_counter()
        return InferenceResult(
            output=prompt,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(prompt.split()),
        )


class OpenAICompatibleProvider:
    """Chat-completions client for any OpenAI-compatible local backend.

    Speaks the OpenAI wire format directly over httpx; ``base_url`` and ``model`` select the
    backend and model. Generation parameters (``temperature``, ``max_tokens``, ...) come from
    the per-call config.
    """

    _GENERATION_KEYS = ("temperature", "max_tokens", "top_p", "stop", "seed")

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.model = model

    def generate(self, prompt: str, config: dict[str, Any]) -> InferenceResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update({k: config[k] for k in self._GENERATION_KEYS if k in config})
        start = time.perf_counter()
        try:
            resp = self._http.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise InferenceError(f"inference request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            raise InferenceError(f"inference backend returned HTTP {resp.status_code}")
        data = resp.json()
        try:
            output = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"malformed inference response: {exc}") from exc
        usage = data.get("usage") or {}
        return InferenceResult(
            output=output,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def get_provider(name: str, *, model: str, config: dict[str, Any]) -> InferenceProvider:
    """Build the named provider. ``config`` may override ``base_url``/``model``/``api_key``."""
    if name == "echo":
        return EchoProvider()
    if name == "openai":
        return OpenAICompatibleProvider(
            base_url=config.get("base_url", settings.serving_url),
            model=config.get("model", model),
            api_key=config.get("api_key"),
        )
    raise InferenceError(f"unknown inference provider: {name!r}")
