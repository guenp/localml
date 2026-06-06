"""Model-version lifecycle state machine.

States and transitions (see Appendix B of the design doc)::

    created → candidate → staging → production → deprecated → archived
    candidate/staging → failed
    created/candidate/failed/deprecated → archived

``failed`` and ``archived`` are terminal-ish; ``archived`` is fully terminal.
"""

from __future__ import annotations

CREATED = "created"
CANDIDATE = "candidate"
STAGING = "staging"
PRODUCTION = "production"
FAILED = "failed"
DEPRECATED = "deprecated"
ARCHIVED = "archived"

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    CREATED: {CANDIDATE, ARCHIVED},
    CANDIDATE: {STAGING, FAILED, ARCHIVED},
    STAGING: {PRODUCTION, FAILED, ARCHIVED},
    PRODUCTION: {DEPRECATED},
    FAILED: {ARCHIVED},
    DEPRECATED: {ARCHIVED},
    ARCHIVED: set(),
}

# Lifecycle states from which a model version may be deployed.
DEPLOYABLE = {STAGING, PRODUCTION}


def can_transition(current: str, target: str) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, set())


class InvalidTransition(Exception):
    """Raised when a lifecycle transition is not permitted."""

    def __init__(self, current: str, target: str) -> None:
        super().__init__(f"cannot transition model version from '{current}' to '{target}'")
        self.current = current
        self.target = target
