"""PostgreSQL connection management for RiskForge persistence."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from riskforge.persistence.exceptions import (
    PersistenceConnectionError,
)

logger = logging.getLogger("riskforge.persistence.postgres.connection")

# ---------------------------------------------------------------------------
# Circuit breaker state
# ---------------------------------------------------------------------------
_CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before opening
_CIRCUIT_BREAKER_COOLDOWN_S = 30.0  # seconds to wait before half-open


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
    min_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("PGMINPOOL", "1"))
    )
    max_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("PGMAXPOOL", "5"))
    )

    def dsn(self) -> str:
        """Return a libpq-style DSN string."""
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.dbname}",
            f"user={self.user}",
            f"connect_timeout={self.connect_timeout}",
        ]
        if self.password:
            parts.append("password=***")
        return " ".join(parts)

    def _dsn_full(self) -> str:
        """Return the full DSN including password (internal use only)."""
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.dbname}",
            f"user={self.user}",
            f"connect_timeout={self.connect_timeout}",
        ]
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)


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
                )
                logger.info(
                    "postgresql pool created",
                    extra={
                        "host": self._config.host,
                        "port": self._config.port,
                        "dbname": self._config.dbname,
                        "min_pool_size": self._config.min_pool_size,
                        "max_pool_size": self._config.max_pool_size,
                    },
                )
            except Exception as exc:
                self._record_failure()
                raise PersistenceConnectionError(
                    "cannot create PostgreSQL connection pool"
                ) from exc
        return self._pool

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
                conn = self._get_pool().getconn()
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

    def close(self) -> None:
        """Shut down the connection pool."""
        if self._pool is not None:
            logger.info("closing postgresql pool")
            self._pool.close()
            self._pool = None
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
