"""Structured JSON logging configuration with correlation ID propagation.

Provides a single :func:`configure_logging` entry point that sets up
Python's standard library ``logging`` with a JSON formatter, context-aware
correlation ID propagation, and automatic masking of sensitive fields.

Usage::

    from riskforge.logging_config import configure_logging
    configure_logging()  # reads RISKFORGE_LOG_LEVEL env var, default INFO
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Correlation ID context variable
# ---------------------------------------------------------------------------

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# ---------------------------------------------------------------------------
# Sensitive field masking
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"password\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
]

_MASKED_VALUE = "***"

_REDACT_KEYS = frozenset({
    "password",
    "dsn",
    "authorization",
    "token",
    "secret",
    "api_key",
    "apikey",
})


def _mask_sensitive(text: str) -> str:
    """Replace sensitive patterns (passwords, bearer tokens) in a string."""
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(_MASKED_VALUE, text)
    return text


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove or mask sensitive fields from a log record."""
    sanitized: dict[str, Any] = {}
    for key, value in record.items():
        lower_key = key.lower()
        if lower_key in _REDACT_KEYS:
            sanitized[key] = _MASKED_VALUE
        elif isinstance(value, str):
            sanitized[key] = _mask_sensitive(value)
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_record(value)
        else:
            sanitized[key] = value
    return sanitized


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter with correlation ID injection."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject correlation ID from context variable
        corr_id = correlation_id_var.get()
        if corr_id is not None:
            log_entry["correlation_id"] = corr_id

        # Include source location for WARNING and above
        if record.levelno >= logging.WARNING:
            log_entry["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
            if record.exc_text:
                log_entry["exception"]["traceback"] = record.exc_text

        # Include any extra fields passed via logger.info("msg", extra={...})
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__  # exclude standard fields
            and k not in {
                "message",
                "msg",
                "args",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "filename",
                "module",
                "levelname",
                "levelno",
                "pathname",
                "process",
                "processName",
                "thread",
                "threadName",
                "name",
                "created",
                "msecs",
                "relativeCreated",
                "taskName",
            }
        }
        if extras:
            log_entry["extra"] = _sanitize_record(extras)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_logging(
    level: str | int | None = None,
    stream: Any | None = None,
) -> None:
    """Configure structured JSON logging for the RiskForge application.

    Parameters
    ----------
    level:
        Log level name or int.  Falls back to the ``RISKFORGE_LOG_LEVEL``
        environment variable, then to ``"INFO"``.
    stream:
        Output stream for the handler.  Defaults to ``sys.stderr``.
    """
    if level is None:
        level = os.environ.get("RISKFORGE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    for existing in root_logger.handlers[:]:
        root_logger.removeHandler(existing)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Suppress noisy third-party loggers
    for name in ("httpcore", "httpx", "uvicorn.access", "psycopg_pool"):
        logging.getLogger(name).setLevel(logging.WARNING)
