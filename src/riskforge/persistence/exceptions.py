"""Persistence boundary exceptions."""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Base class for persistence failures."""


class PersistenceConnectionError(PersistenceError):
    """Raised when the persistence backend is unreachable or the pool is exhausted."""

    def __init__(self, message: str = "persistence connection unavailable") -> None:
        super().__init__(message)


class PersistenceTimeoutError(PersistenceError):
    """Raised when a persistence operation exceeds its time limit."""

    def __init__(self, message: str = "persistence operation timed out") -> None:
        super().__init__(message)


class PersistenceConflictError(PersistenceError):
    """Raised when an operation conflicts with an existing immutable record."""


class RecordNotFoundError(PersistenceError):
    """Raised when a requested persistent record does not exist."""


class InvalidPageError(PersistenceError):
    """Raised when pagination parameters are invalid."""


class InvalidFilterError(PersistenceError):
    """Raised when repository filter values are invalid."""
