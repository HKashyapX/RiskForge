"""PostgreSQL connection management for RiskForge persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from riskforge.persistence.exceptions import PersistenceError


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
            parts.append(f"password={self.password}")
        return " ".join(parts)


class PostgresConnectionPool:
    """Manages a psycopg connection pool.

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

    def _get_pool(self):  # type: ignore[no-untyped-def]
        """Lazily create the connection pool."""
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            try:
                self._pool = ConnectionPool(
                    conninfo=self._config.dsn(),
                    min_size=self._config.min_pool_size,
                    max_size=self._config.max_pool_size,
                    check=ConnectionPool.check_connection,
                )
            except Exception as exc:
                raise PersistenceError(
                    "cannot create PostgreSQL connection pool"
                ) from exc
        return self._pool

    def getconn(self):  # type: ignore[no-untyped-def]
        """Acquire a connection from the pool."""
        try:
            return self._get_pool().getconn()
        except Exception as exc:
            raise PersistenceError("cannot acquire PostgreSQL connection") from exc

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
            self._pool.close()
            self._pool = None
