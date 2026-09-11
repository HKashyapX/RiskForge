"""Unit tests for PostgreSQL connection configuration.

These tests never touch a live database: they validate environment parsing,
configuration invariants, DSN composition, timeout classification, and the
pool readiness snapshot shape.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from riskforge.persistence.postgres.connection import (
    PoolReadiness,
    PostgresConfig,
    is_timeout_error,
)

_PG_ENV_VARS = [
    "PGHOST",
    "PGPORT",
    "PGDATABASE",
    "PGUSER",
    "PGPASSWORD",
    "PGCONNECT_TIMEOUT",
    "PGSSLMODE",
    "PGMINPOOL",
    "PGMAXPOOL",
    "PGPOOL_TIMEOUT",
    "PGSTATEMENT_TIMEOUT_MS",
]


@pytest.fixture()
def clean_pg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Environment-driven defaults and overrides
# ---------------------------------------------------------------------------


class TestPostgresConfigEnv:
    def test_defaults_with_clean_env(self, clean_pg_env: None) -> None:
        config = PostgresConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.dbname == "riskforge"
        assert config.user == "postgres"
        assert config.password == ""
        assert config.connect_timeout == 10
        assert config.sslmode == "prefer"
        assert config.min_pool_size == 1
        assert config.max_pool_size == 5
        assert config.pool_timeout == 30.0
        assert config.statement_timeout_ms is None

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGHOST", "db.example.com")
        monkeypatch.setenv("PGPORT", "6543")
        monkeypatch.setenv("PGDATABASE", "rf_prod")
        monkeypatch.setenv("PGUSER", "rf_user")
        monkeypatch.setenv("PGPASSWORD", "s3cret")
        monkeypatch.setenv("PGCONNECT_TIMEOUT", "3")
        monkeypatch.setenv("PGSSLMODE", "verify-full")
        monkeypatch.setenv("PGMINPOOL", "2")
        monkeypatch.setenv("PGMAXPOOL", "8")
        monkeypatch.setenv("PGPOOL_TIMEOUT", "5.5")
        monkeypatch.setenv("PGSTATEMENT_TIMEOUT_MS", "2500")
        config = PostgresConfig()
        assert config.host == "db.example.com"
        assert config.port == 6543
        assert config.sslmode == "verify-full"
        assert config.min_pool_size == 2
        assert config.max_pool_size == 8
        assert config.pool_timeout == 5.5
        assert config.statement_timeout_ms == 2500

    def test_empty_optional_vars_are_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGPOOL_TIMEOUT", "")
        monkeypatch.setenv("PGSTATEMENT_TIMEOUT_MS", "")
        config = PostgresConfig()
        assert config.pool_timeout == 30.0
        assert config.statement_timeout_ms is None


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestPostgresConfigValidation:
    def test_invalid_sslmode_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGSSLMODE"):
            PostgresConfig(sslmode="bogus")

    def test_all_valid_ssl_modes_accepted(self) -> None:
        for mode in ("disable", "allow", "prefer", "require", "verify-ca", "verify-full"):
            assert PostgresConfig(sslmode=mode).sslmode == mode

    def test_port_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPORT"):
            PostgresConfig(port=0)

    def test_port_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPORT"):
            PostgresConfig(port=-1)

    def test_port_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPORT"):
            PostgresConfig(port=65536)

    def test_port_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPORT"):
            PostgresConfig(port=True)

    def test_negative_connect_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGCONNECT_TIMEOUT"):
            PostgresConfig(connect_timeout=-1)

    def test_bool_connect_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGCONNECT_TIMEOUT"):
            PostgresConfig(connect_timeout=True)

    def test_negative_min_pool_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGMINPOOL"):
            PostgresConfig(min_pool_size=-1)

    def test_zero_max_pool_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGMAXPOOL"):
            PostgresConfig(max_pool_size=0)

    def test_bool_pool_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGMINPOOL"):
            PostgresConfig(min_pool_size=True)

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGMINPOOL must not exceed"):
            PostgresConfig(min_pool_size=5, max_pool_size=3)

    def test_zero_pool_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPOOL_TIMEOUT"):
            PostgresConfig(pool_timeout=0.0)

    def test_negative_pool_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPOOL_TIMEOUT"):
            PostgresConfig(pool_timeout=-1.0)

    def test_nan_pool_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPOOL_TIMEOUT"):
            PostgresConfig(pool_timeout=math.nan)

    def test_inf_pool_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGPOOL_TIMEOUT"):
            PostgresConfig(pool_timeout=math.inf)

    def test_negative_statement_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGSTATEMENT_TIMEOUT_MS"):
            PostgresConfig(statement_timeout_ms=-1)

    def test_bool_statement_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="PGSTATEMENT_TIMEOUT_MS"):
            PostgresConfig(statement_timeout_ms=True)


# ---------------------------------------------------------------------------
# DSN composition
# ---------------------------------------------------------------------------


class TestPostgresConfigDsn:
    def test_dsn_includes_sslmode(self) -> None:
        config = PostgresConfig(sslmode="require")
        assert "sslmode=require" in config.dsn()

    def test_dsn_includes_connect_timeout(self) -> None:
        config = PostgresConfig(connect_timeout=7)
        assert "connect_timeout=7" in config.dsn()

    def test_dsn_masks_password(self) -> None:
        config = PostgresConfig(password="supersecret")
        dsn = config.dsn()
        assert "supersecret" not in dsn
        assert "password=***" in dsn

    def test_dsn_full_includes_password(self) -> None:
        config = PostgresConfig(password="supersecret")
        dsn = config._dsn_full()
        assert "supersecret" in dsn

    def test_dsn_full_includes_sslmode(self) -> None:
        config = PostgresConfig(sslmode="verify-full")
        assert "sslmode=verify-full" in config._dsn_full()

    def test_dsn_no_statement_options_when_unset(self) -> None:
        config = PostgresConfig(statement_timeout_ms=None)
        assert "statement_timeout" not in config.dsn()

    def test_dsn_no_statement_options_when_zero(self) -> None:
        config = PostgresConfig(statement_timeout_ms=0)
        assert "statement_timeout" not in config.dsn()

    def test_dsn_quotes_statement_option(self) -> None:
        config = PostgresConfig(statement_timeout_ms=5000)
        assert "options='-c statement_timeout=5000'" in config.dsn()
        assert "options='-c statement_timeout=5000'" in config._dsn_full()


# ---------------------------------------------------------------------------
# Timeout classification
# ---------------------------------------------------------------------------


class TestIsTimeoutError:
    def test_stdlib_timeout_error(self) -> None:
        assert is_timeout_error(TimeoutError("timed out")) is True

    def test_pool_timeout_error(self) -> None:
        from psycopg_pool import PoolTimeout

        assert is_timeout_error(PoolTimeout("pool timeout")) is True

    def test_psycopg_query_canceled(self) -> None:
        from psycopg.errors import QueryCanceled

        assert is_timeout_error(QueryCanceled("canceling statement due to statement timeout")) is True

    def test_message_fallback_timeout(self) -> None:
        class _FakeTimeout(Exception):
            pass

        assert is_timeout_error(_FakeTimeout("connection timed out")) is True

    def test_generic_error_is_not_timeout(self) -> None:
        assert is_timeout_error(RuntimeError("connection refused")) is False

    def test_connection_error_is_not_timeout(self) -> None:
        assert is_timeout_error(ConnectionError("no route to host")) is False

    def test_not_an_exception_yielding_object(self) -> None:
        assert is_timeout_error(KeyError("nope")) is False


# ---------------------------------------------------------------------------
# Pool readiness snapshot
# ---------------------------------------------------------------------------


class TestPoolReadiness:
    def test_defaults(self) -> None:
        snapshot = PoolReadiness()
        assert snapshot.name == "postgres_pool"
        assert snapshot.ready is False
        assert snapshot.detail is None

    def test_explicit_values(self) -> None:
        snapshot = PoolReadiness("postgres_pool", True, "pool ready")
        assert snapshot.ready is True
        assert snapshot.detail == "pool ready"

    def test_frozen(self) -> None:
        snapshot = PoolReadiness()
        with pytest.raises(FrozenInstanceError):
            snapshot.ready = True  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PoolReadiness(name="")

    def test_overlong_detail_rejected(self) -> None:
        with pytest.raises(ValueError, match="detail"):
            PoolReadiness(detail="x" * 201)