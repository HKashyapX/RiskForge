"""PostgreSQL connection management for RiskForge persistence."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field

from riskforge.persistence.exceptions import (
    PersistenceConnectionError,
    PersistenceTimeoutError,
)

logger = logging.getLogger("riskforge.persistence.postgres.connection")

# ---------------------------------------------------------------------------
# Circuit breaker state
# ---------------------------------------------------------------------------
_CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before opening
_CIRCUIT_BREAKER_COOLDOWN_S = 30.0  # seconds to wait before half-open

# ---------------------------------------------------------------------------
# libpq sslmode values
# ---------------------------------------------------------------------------
_VALID_SSL_MODES = frozenset(
    {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
)

# Stable component name used by runtime lifecycle/readiness composition.
_POOL_COMPONENT_NAME = "postgres_pool"


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable, treating unset/empty as default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_optional_int(name: str) -> int | None:
    """Parse an optional integer environment variable; unset/empty -> None."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def is_timeout_error(exc: Exception) -> bool:
    """Classify an exception as a database/pool timeout.

    Recognises psycopg and psycopg_pool timeout exceptions when available,
    plus standard-library ``TimeoutError`` and a defensive string fallback for
    environments where the libraries are not importable.
    """
    if isinstance(exc, TimeoutError):
        return True
    for module_name, error_name in (
        ("psycopg.errors", "QueryCanceled"),
        ("psycopg_pool", "PoolTimeout"),
    ):
        try:
            module = __import__(module_name, fromlist=[error_name])
            error_type = getattr(module, error_name)
            if isinstance(exc, error_type):
                return True
        except Exception:  # noqa: BLE001, S112 — optional classification, never masks
            continue
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


@dataclass(frozen=True)
class PostgresConfig:
    """Immutable PostgreSQL connection configuration.

    All values are read from environment variables with sensible defaults
    for local development.  Production deployments (e.g. Supabase) inject
    the environment variables directly.
    """

    host: str = field(default_factory=lambda: os.environ.get("PGHOST", "localhost"))
    port: int = field(
        default_factory=lambda: int(os.environ.get("PGPORT", "5432"))
    )
    dbname: str = field(default_factory=lambda: os.environ.get("PGDATABASE", "riskforge"))
    user: str = field(default_factory=lambda: os.environ.get("PGUSER", "postgres"))
    password: str = field(default_factory=lambda: os.environ.get("PGPASSWORD", ""))
    connect_timeout: int = field(
        default_factory=lambda: int(os.environ.get("PGCONNECT_TIMEOUT", "10"))
    )
    sslmode: str = field(
        default_factory=lambda: os.environ.get("PGSSLMODE", "prefer")
    )
    min_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("PGMINPOOL", "1"))
    )
    max_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("PGMAXPOOL", "5"))
    )
    pool_timeout: float = field(
        default_factory=lambda: _env_float("PGPOOL_TIMEOUT", 30.0)
    )
    statement_timeout_ms: int | None = field(
        default_factory=lambda: _env_optional_int("PGSTATEMENT_TIMEOUT_MS")
    )

    def __post_init__(self) -> None:
        if self.sslmode not in _VALID_SSL_MODES:
            raise ValueError(
                "PGSSLMODE must be one of: disable, allow, prefer, require, "
                "verify-ca, verify-full"
            )
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("PGPORT must be an integer between 1 and 65535")
        if (
            isinstance(self.connect_timeout, bool)
            or not isinstance(self.connect_timeout, int)
            or self.connect_timeout < 0
        ):
            raise ValueError("PGCONNECT_TIMEOUT must be a non-negative integer")
        if (
            isinstance(self.min_pool_size, bool)
            or not isinstance(self.min_pool_size, int)
            or self.min_pool_size < 0
        ):
            raise ValueError("PGMINPOOL must be a non-negative integer")
        if (
            isinstance(self.max_pool_size, bool)
            or not isinstance(self.max_pool_size, int)
            or self.max_pool_size < 1
        ):
            raise ValueError("PGMAXPOOL must be a positive integer")
        if self.min_pool_size > self.max_pool_size:
            raise ValueError("PGMINPOOL must not exceed PGMAXPOOL")
        if not math.isfinite(self.pool_timeout) or self.pool_timeout <= 0:
            raise ValueError("PGPOOL_TIMEOUT must be a finite number greater than 0")
        if self.statement_timeout_ms is not None and (
            isinstance(self.statement_timeout_ms, bool)
            or not isinstance(self.statement_timeout_ms, int)
            or self.statement_timeout_ms < 0
        ):
            raise ValueError(
                "PGSTATEMENT_TIMEOUT_MS must be a non-negative integer"
            )

    def _dsn_parts(self) -> list[str]:
        """Shared libpq DSN keyword/value pairs (password never included)."""
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.dbname}",
            f"user={self.user}",
            f"connect_timeout={self.connect_timeout}",
            f"sslmode={self.sslmode}",
        ]
        if self.statement_timeout_ms is not None and self.statement_timeout_ms > 0:
            parts.append(
                f"options='-c statement_timeout={self.statement_timeout_ms}'"
            )
        return parts

    def dsn(self) -> str:
        """Return a libpq-style DSN string (password masked)."""
        parts = self._dsn_parts()
        if self.password:
            parts.append("password=***")
        return " ".join(parts)

    def _dsn_full(self) -> str:
        """Return the full DSN including password (internal use only)."""
        parts = self._dsn_parts()
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)


@dataclass(frozen=True)
class PoolReadiness:
    """Transport-neutral readiness snapshot for the PostgreSQL pool.

    The field shape mirrors the runtime :class:`ComponentReadiness` contract
    so the pool can be adapted by the runtime composer without coupling
    persistence to the runtime subsystem.
    """

    name: str = _POOL_COMPONENT_NAME
    ready: bool = False
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("readiness name must not be empty")
        if self.detail is not None and len(self.detail) > 200:
            raise ValueError("readiness detail is too long")


class PostgresConnectionPool:
    """Manages a psycopg connection pool with circuit breaker protection.

    This class handles connection pooling only.  Schema management is
    performed by :func:`riskforge.persistence.postgres.migrate.run_migrations`,
    which must be executed **before** the application uses the repositories.

    Parameters
    ----------
    config:
        Connection configuration.  Defaults to environment-based values.
    """

    def __init__(self, config: PostgresConfig | None = None) -> None:
        self._config = config or PostgresConfig()
        self._pool = None  # type: ignore[type-arg]
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0  # monotonic timestamp

    @property
    def name(self) -> str:
        """Return the stable lifecycle component name."""
        return _POOL_COMPONENT_NAME

    def _is_circuit_open(self) -> bool:
        """Check if the circuit breaker is currently open."""
        if self._consecutive_failures < _CIRCUIT_BREAKER_THRESHOLD:
            return False
        return time.monotonic() < self._circuit_open_until

    def _record_success(self) -> None:
        """Reset circuit breaker on successful connection."""
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Track consecutive failures and open circuit when threshold reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN_S
            logger.warning(
                "circuit breaker opened after %d consecutive failures",
                self._consecutive_failures,
            )

    def _get_pool(self):  # type: ignore[no-untyped-def]
        """Lazily create the connection pool."""
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            try:
                self._pool = ConnectionPool(
                    conninfo=self._config._dsn_full(),
                    min_size=self._config.min_pool_size,
                    max_size=self._config.max_pool_size,
                    check=ConnectionPool.check_connection,
                    timeout=self._config.pool_timeout,
                )
                logger.info(
                    "postgresql pool created",
                    extra={
                        "host": self._config.host,
                        "port": self._config.port,
                        "dbname": self._config.dbname,
                        "sslmode": self._config.sslmode,
                        "min_pool_size": self._config.min_pool_size,
                        "max_pool_size": self._config.max_pool_size,
                        "pool_timeout": self._config.pool_timeout,
                    },
                )
            except Exception as exc:
                self._record_failure()
                raise PersistenceConnectionError(
                    "cannot create PostgreSQL connection pool"
                ) from exc
        return self._pool

    def open(self) -> None:
        """Create the pool and wait until its minimum connections are ready.

        Fail-fast variant of the lazy ``_get_pool()`` path: used by the
        runtime lifecycle so an unreachable database blocks startup instead of
        failing on the first request.  Idempotent: a pool that is already
        created and ready returns immediately.
        """
        pool = self._get_pool()
        try:
            pool.wait(timeout=self._config.pool_timeout)
        except Exception as exc:
            self._record_failure()
            if is_timeout_error(exc):
                raise PersistenceTimeoutError(
                    "PostgreSQL pool did not become ready within the configured timeout"
                ) from exc
            raise PersistenceConnectionError(
                "cannot open PostgreSQL connection pool"
            ) from exc

    def getconn(self):  # type: ignore[no-untyped-def]
        """Acquire a connection from the pool."""
        if self._is_circuit_open():
            raise PersistenceConnectionError(
                "PostgreSQL circuit breaker is open; connection refused"
            )

        retries = 2
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                conn = self._get_pool().getconn(timeout=self._config.pool_timeout)
                self._record_success()
                return conn
            except PersistenceConnectionError:
                raise
            except Exception as exc:  # noqa: BLE001 — catch-all for retry logic
                last_exc = exc
                self._record_failure()
                if attempt < retries:
                    wait = 0.1 * (2**attempt)  # exponential backoff: 0.1s, 0.2s
                    logger.warning(
                        "connection attempt %d/%d failed, retrying in %.1fs",
                        attempt + 1,
                        retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                break

        if last_exc is not None and is_timeout_error(last_exc):
            raise PersistenceTimeoutError(
                "PostgreSQL connection acquisition timed out"
            ) from last_exc
        raise PersistenceConnectionError(
            "cannot acquire PostgreSQL connection"
        ) from last_exc

    def putconn(self, conn) -> None:  # type: ignore[no-untyped-def]
        """Return a connection to the pool."""
        if self._pool is not None:
            try:
                self._pool.putconn(conn)
            except Exception:  # noqa: BLE001, S110 — connection may already be returned
                pass

    def readiness(self) -> PoolReadiness:
        """Return a lightweight, read-only readiness snapshot.

        The probe never mutates circuit breaker state: it only observes the
        pool and, when a pool exists, verifies that a connection can be
        acquired and returned.
        """
        if self._pool is None:
            return PoolReadiness(_POOL_COMPONENT_NAME, False, "pool not created")
        if self._is_circuit_open():
            return PoolReadiness(_POOL_COMPONENT_NAME, False, "circuit breaker open")
        try:
            conn = self._pool.getconn(timeout=self._config.pool_timeout)
        except Exception as exc:  # noqa: BLE001 — isolate arbitrary probe failures
            if is_timeout_error(exc):
                return PoolReadiness(
                    _POOL_COMPONENT_NAME, False, "connection acquisition timed out"
                )
            return PoolReadiness(
                _POOL_COMPONENT_NAME, False, "unable to acquire connection"
            )
        try:
            self._pool.putconn(conn)
        except Exception:  # noqa: BLE001, S110 — probe cleanup is best-effort
            pass
        return PoolReadiness(_POOL_COMPONENT_NAME, True, "pool ready")

    def close(self) -> None:
        """Shut down the connection pool idempotently."""
        if self._pool is not None:
            logger.info("closing postgresql pool")
            self._pool.close()
            self._pool = None
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
