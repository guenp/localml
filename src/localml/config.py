"""SDK configuration.

Configuration precedence (highest first):

1. Explicit arguments to :func:`configure`.
2. Environment variables (``LOCALML_API_URL``, ``LOCALML_API_TOKEN``).
3. ``~/.localml/config.toml``.
4. Built-in defaults.

The active config is process-global; the control plane remains the source of truth for
all platform state.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, SupportsIndex

DEFAULT_API_URL = "http://localhost:8000"
CONFIG_PATH = Path.home() / ".localml" / "config.toml"


@dataclass
class Config:
    """Resolved SDK configuration."""

    api_url: str = DEFAULT_API_URL
    token: str | None = None
    timeout: float = 30.0
    max_retries: int = 3


_active: Config | None = None


def _load_file(path: Path = CONFIG_PATH) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _float_config(value: object, default: float) -> float:
    if isinstance(value, str | bytes | bytearray | SupportsFloat | SupportsIndex):
        return float(value)
    return default


def _int_config(value: object, default: int) -> int:
    if isinstance(value, str | bytes | bytearray | SupportsIndex):
        return int(value)
    return default


def configure(
    api_url: str | None = None,
    token: str | None = None,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Config:
    """Set and return the active SDK configuration.

    Values not provided fall back to environment variables, then the config file, then
    built-in defaults.
    """
    global _active
    file_cfg = _load_file()

    cfg = Config(
        api_url=(
            api_url
            or os.environ.get("LOCALML_API_URL")
            or str(file_cfg.get("api_url", DEFAULT_API_URL))
        ),
        token=(
            token
            or os.environ.get("LOCALML_API_TOKEN")
            or (str(file_cfg["token"]) if "token" in file_cfg else None)
        ),
        timeout=timeout if timeout is not None else _float_config(file_cfg.get("timeout"), 30.0),
        max_retries=(
            max_retries if max_retries is not None else _int_config(file_cfg.get("max_retries"), 3)
        ),
    )
    _active = cfg
    return cfg


def get_config() -> Config:
    """Return the active config, initializing from env/file/defaults on first use."""
    if _active is None:
        return configure()
    return _active
