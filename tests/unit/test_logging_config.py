"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging
from io import StringIO

from riskforge.logging_config import (
    JSONFormatter,
    _mask_sensitive,
    _sanitize_record,
    configure_logging,
    correlation_id_var,
)


class TestCorrelationIdVar:
    def test_default_is_none(self) -> None:
        token = correlation_id_var.set(None)
        try:
            assert correlation_id_var.get() is None
        finally:
            correlation_id_var.reset(token)

    def test_set_and_get(self) -> None:
        token = correlation_id_var.set("req-abc-123")
        try:
            assert correlation_id_var.get() == "req-abc-123"
        finally:
            correlation_id_var.reset(token)


class TestMaskSensitive:
    def test_masks_password_in_dsn(self) -> None:
        text = "host=localhost password=supersecret dbname=riskforge"
        masked = _mask_sensitive(text)
        assert "supersecret" not in masked
        assert "***" in masked
        assert "host=localhost" in masked

    def test_masks_bearer_token(self) -> None:
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9"
        masked = _mask_sensitive(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in masked
        assert "***" in masked

    def test_preserves_safe_text(self) -> None:
        text = "request completed in 25ms"
        assert _mask_sensitive(text) == text


class TestSanitizeRecord:
    def test_masks_password_key(self) -> None:
        record = {"host": "localhost", "password": "secret123", "port": 5432}
        sanitized = _sanitize_record(record)
        assert sanitized["password"] == "***"
        assert sanitized["host"] == "localhost"
        assert sanitized["port"] == 5432

    def test_masks_nested_sensitive_keys(self) -> None:
        record = {"db": {"dsn": "host=x password=y", "name": "riskforge"}}
        sanitized = _sanitize_record(record)
        assert sanitized["db"]["dsn"] == "***"
        assert sanitized["db"]["name"] == "riskforge"

    def test_masks_token_key(self) -> None:
        record = {"token": "abc123", "user": "admin"}
        sanitized = _sanitize_record(record)
        assert sanitized["token"] == "***"
        assert sanitized["user"] == "admin"

    def test_masks_authorization_key(self) -> None:
        record = {"authorization": "Bearer xyz"}
        sanitized = _sanitize_record(record)
        assert sanitized["authorization"] == "***"

    def test_masks_api_key(self) -> None:
        for key in ("api_key", "apikey", "API_KEY", "ApiKey"):
            record = {key: "secret-key-value"}
            sanitized = _sanitize_record(record)
            assert sanitized[key] == "***"


class TestJSONFormatter:
    def test_basic_log_entry(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="riskforge.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "riskforge.test"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_includes_correlation_id(self) -> None:
        formatter = JSONFormatter()
        token = correlation_id_var.set("req-xyz-789")
        try:
            record = logging.LogRecord(
                name="riskforge.test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["correlation_id"] == "req-xyz-789"
        finally:
            correlation_id_var.reset(token)

    def test_warning_includes_source(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="riskforge.test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=99,
            msg="warning msg",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "WARNING"
        assert "source" in parsed
        assert parsed["source"]["file"] == "test.py"
        assert parsed["source"]["line"] == 99

    def test_debug_does_not_include_source(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="riskforge.test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="debug msg",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "source" not in parsed

    def test_exception_info_included(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("bad value")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="riskforge.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "bad value"

    def test_extra_fields_included(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="riskforge.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="with extras",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"  # type: ignore[attr-defined]
        record.batch_size = 16  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        # Extra fields should be present (non-standard LogRecord fields)
        assert "extra" in parsed


class TestConfigureLogging:
    def test_configures_root_logger(self) -> None:
        stream = StringIO()
        configure_logging(level="DEBUG", stream=stream)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1

    def test_default_level_is_info(self) -> None:
        import os

        os.environ.pop("RISKFORGE_LOG_LEVEL", None)
        stream = StringIO()
        configure_logging(stream=stream)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_suppresses_noisy_loggers(self) -> None:
        configure_logging()
        for name in ("httpcore", "httpx", "uvicorn.access", "psycopg_pool"):
            logger = logging.getLogger(name)
            assert logger.level >= logging.WARNING
