"""Custom serving-backend registry (Phase 4).

Serving is an OpenAI-compatible proxy: the control plane forwards ``/v1/chat/completions``
to a backend ``{base_url, model, api_key}``. Ollama, MLX-LM, llama.cpp, and vLLM all speak
that API out of the box, so they need no registration — point a deployment at their URL.

This registry names a backend so deployments can refer to it by a short handle instead of
repeating connection details:

    ml.providers.register("my-ollama", base_url="http://gpu-box:11434", model="llama3")
    ml.deploy(version, provider="my-ollama")

The registered spec is expanded into the deployment's ``config`` (backend overrides resolved
at proxy time), so a swap to another provider is just another ``deploy``/``swap`` call.
"""

from __future__ import annotations

from typing import Any


class Provider:
    """A named OpenAI-compatible serving backend."""

    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def as_config(self) -> dict[str, Any]:
        """The deployment ``config`` overrides for this backend (omitting unset fields)."""
        config: dict[str, Any] = {"base_url": self.base_url}
        if self.model is not None:
            config["model"] = self.model
        if self.api_key is not None:
            config["api_key"] = self.api_key
        return config


_PROVIDERS: dict[str, Provider] = {}


def register(
    name: str, *, base_url: str, model: str | None = None, api_key: str | None = None
) -> Provider:
    """Register a named serving backend for use as ``ml.deploy(..., provider=name)``."""
    provider = Provider(base_url=base_url, model=model, api_key=api_key)
    _PROVIDERS[name] = provider
    return provider


def get(name: str) -> Provider:
    """Return a registered provider, or raise ``KeyError`` if unknown."""
    if name not in _PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; register it with ml.providers.register")
    return _PROVIDERS[name]


def config_for(name: str) -> dict[str, Any]:
    """Return the deployment-config overrides for a registered provider."""
    return get(name).as_config()
