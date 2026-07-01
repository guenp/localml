"""Content hashing for artifact staging.

A stable checksum lets the control plane deduplicate and verify artifacts, and is recorded
alongside the registry entry so a later download can be integrity-checked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 16


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
