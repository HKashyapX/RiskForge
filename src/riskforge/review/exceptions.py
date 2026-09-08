"""Review workflow error taxonomy."""

from __future__ import annotations


class ReviewWorkflowError(RuntimeError):
    """Base class for review workflow failures."""


class ReviewNotFoundError(ReviewWorkflowError):
    """Raised when a requested incident has no persisted automated result."""


class ReviewPermissionError(ReviewWorkflowError):
    """Raised when a reviewer is not authorized for an action."""


class ReviewConflictError(ReviewWorkflowError):
    """Raised when a decision ID or audit write conflicts with existing history."""


class ReviewPersistenceError(ReviewWorkflowError):
    """Raised when the atomic review/audit writer fails."""
