"""Typed SDK exceptions.

All SDK errors derive from :class:`LocalMLError` so callers can catch the whole family
with a single ``except`` while still being able to discriminate specific failures.
"""

from __future__ import annotations


class LocalMLError(Exception):
    """Base class for all localml SDK errors."""


class AuthenticationError(LocalMLError):
    """Raised when the API rejects the configured token (HTTP 401)."""


class ValidationError(LocalMLError):
    """Raised for invalid arguments or rejected requests (HTTP 400/422)."""


class ArtifactUploadError(LocalMLError):
    """Raised when an artifact upload fails or is incomplete."""


class ModelRegistrationError(LocalMLError):
    """Raised when registering a model version fails."""


class EvaluationFailedError(LocalMLError):
    """Raised when an evaluation job ends in a ``failed`` state."""


class DeploymentError(LocalMLError):
    """Raised when a deployment cannot be created or activated."""
