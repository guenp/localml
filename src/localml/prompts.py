"""Prompt registry helpers.

Prompts are versioned ``str.format`` templates with a restricted, sandboxed field grammar
(bare identifiers only — the server rejects attribute/index access at registration).
Variables are extracted server-side; rendering requires exactly the declared variables.
"""

from __future__ import annotations

from typing import Any

from .client import get_client
from .types import PromptVersion


def register(
    *,
    name: str,
    template: str,
    project: str = "local",
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PromptVersion:
    """Register a prompt template and return the created version."""
    return get_client().register_prompt(
        name=name, template=template, project=project, version=version, metadata=metadata
    )


def get(name: str) -> list[PromptVersion]:
    """Return all registered versions for a prompt name."""
    return get_client().get_prompts(name)


def render(name: str, version: str, **variables: Any) -> str:
    """Render a registered prompt version server-side."""
    return get_client().render_prompt(name, version, variables)
