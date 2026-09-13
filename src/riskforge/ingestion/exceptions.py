"""Ingestion-specific failures translated to safe API errors."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class UnsupportedFormatError(IngestionError):
    """The requested source format has no parser."""


class ReportParseError(IngestionError):
    """A source document could not be parsed (corrupt, wrong layout)."""


class ResourceLimitError(IngestionError):
    """A source document exceeded a format-aware resource limit.

    Raised before parsing so hostile inputs (row bombs, spreadsheet
    expansion bombs, decompression bombs, page bombs) are refused without
    consuming unbounded memory or CPU.
    """


class ReportValidationError(IngestionError):
    """One or more parsed reports failed semantic validation."""

    def __init__(self, message: str, errors: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []
