"""Unit tests for the lifecycle state machine."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "api"))

from app.lifecycle import (  # noqa: E402
    CANDIDATE,
    CREATED,
    DEPLOYABLE,
    PRODUCTION,
    STAGING,
    can_transition,
)


def test_valid_path():
    assert can_transition(CREATED, CANDIDATE)
    assert can_transition(CANDIDATE, STAGING)
    assert can_transition(STAGING, PRODUCTION)


def test_invalid_skips():
    assert not can_transition(CREATED, PRODUCTION)
    assert not can_transition(CREATED, STAGING)


def test_archived_is_terminal():
    assert not can_transition("archived", CANDIDATE)


def test_deployable_states():
    assert {STAGING, PRODUCTION} == DEPLOYABLE
