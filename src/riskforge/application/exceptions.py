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


class IncidentNotFoundError(ApplicationError):
    """Raised when an incident does not exist."""


class QueryApplicationError(ApplicationError):
    """Raised when incident or audit querying fails."""


class ReviewNotFoundApplicationError(ApplicationError):
    """Raised when a review target does not exist."""


class ReviewPermissionApplicationError(ApplicationError):
    """Raised when a reviewer is not permitted to decide."""


class ReviewConflictApplicationError(ApplicationError):
    """Raised when a review decision conflicts with existing history."""


class ReviewApplicationError(ApplicationError):
    """Raised when review processing otherwise fails."""
