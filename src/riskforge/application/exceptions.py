"""Safe application-layer error taxonomy."""

from __future__ import annotations


class ApplicationError(RuntimeError):
    """Base class for application orchestration failures."""


class DuplicateLogIdError(ApplicationError):
    """Raised before inference when a batch repeats a correlation identifier."""


class InferenceApplicationError(ApplicationError):
    """Raised when an inference dependency fails across the application boundary."""


class ResultCorrelationError(ApplicationError):
    """Raised when an inference dependency returns an invalid correlation set."""


class MetricsApplicationError(ApplicationError):
    """Raised when an application metrics dependency fails."""
