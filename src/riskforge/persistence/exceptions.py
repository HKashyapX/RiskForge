"""Persistence boundary exceptions."""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Base class for persistence failures."""


class PersistenceConflictError(PersistenceError):
    """Raised when an operation conflicts with an existing immutable record."""


class RecordNotFoundError(PersistenceError):
    """Raised when a requested persistent record does not exist."""


class InvalidPageError(PersistenceError):
    """Raised when pagination parameters are invalid."""


class InvalidFilterError(PersistenceError):
    """Raised when repository filter values are invalid."""
