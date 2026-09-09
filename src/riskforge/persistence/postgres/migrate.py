"""Versioned schema migration runner for PostgreSQL.

Usage
-----
Run all pending migrations against a live database::

    from riskforge.persistence.postgres.migrate import run_migrations
    run_migrations(dsn="host=localhost dbname=riskforge user=postgres")

Or from the command line::

    python -m riskforge.persistence.postgres.migrate

Both forms use environment variables (PGHOST, PGPORT, PGDATABASE,
PGUSER, PGPASSWORD) when the ``dsn`` parameter is omitted.

Design
------
- Migration files live in ``migrations/`` beside this module.
- Files are sorted lexicographically and executed in order.
- A ``schema_migrations`` table records which migrations have been applied,
  their SHA-256 checksum, and the timestamp of application.
- Each migration is executed inside a single transaction.  If any statement
  fails the entire migration is rolled back and ``run_migrations`` raises
  ``MigrationError``.
- The ``schema_migrations`` table is created automatically on first run.
- Repeated execution of ``run_migrations`` is safe: already-applied
  migrations are skipped.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

from riskforge.persistence.exceptions import PersistenceError

logger = logging.getLogger("riskforge.persistence.postgres.migrate")

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_TRACKING_TABLE = "schema_migrations"


class MigrationError(PersistenceError):
    """Raised when a migration cannot be applied."""


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _checksum(sql_text: str) -> str:
    """Deterministic SHA-256 hex digest of migration content."""
    return hashlib.sha256(sql_text.encode("utf-8")).hexdigest()


def _discover_migrations() -> list[tuple[str, str]]:
    """Return sorted list of (version, sql_text) from the migrations dir."""
    if not _MIGRATIONS_DIR.is_dir():
        return []
    migrations: list[tuple[str, str]] = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = path.stem  # e.g. "001_initial"
        sql_text = path.read_text(encoding="utf-8")
        migrations.append((version, sql_text))
    return migrations


def _ensure_tracking_table(conn: object) -> None:
    """Create the schema_migrations tracking table if it does not exist."""
    with conn.cursor() as cur:  # type: ignore[union-attr]
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {_TRACKING_TABLE} ("
            "version     TEXT PRIMARY KEY,"
            "name        TEXT NOT NULL,"
            "checksum    TEXT NOT NULL,"
            "applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )


def _applied_versions(conn: object) -> dict[str, str]:
    """Return {version: checksum} for all already-applied migrations."""
    with conn.cursor() as cur:  # type: ignore[union-attr]
        cur.execute(f"SELECT version, checksum FROM {_TRACKING_TABLE}")
        return {row[0]: row[1] for row in cur.fetchall()}


def _record_migration(conn: object, version: str, name: str, chk: str) -> None:
    """Insert a row into the tracking table."""
    with conn.cursor() as cur:  # type: ignore[union-attr]
        cur.execute(
            f"INSERT INTO {_TRACKING_TABLE} (version, name, checksum) VALUES (%s, %s, %s)",
            (version, name, chk),
        )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run_migrations(*, dsn: str | None = None) -> int:
    """Apply all pending migrations and return the count applied.

    Parameters
    ----------
    dsn:
        A libpq connection string.  When *None*, environment variables
        (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) are used.

    Returns
    -------
    int
        Number of migrations applied in this invocation.

    Raises
    ------
    MigrationError
        If a migration SQL file fails or a checksum mismatch is detected.
    """
    import psycopg

    effective_dsn = dsn or _dsn_from_env()
    migrations = _discover_migrations()
    if not migrations:
        return 0

    logger.info("running migrations", extra={"pending": len(migrations)})
    applied_count = 0
    try:
        with psycopg.connect(effective_dsn) as conn:
            _ensure_tracking_table(conn)
            existing = _applied_versions(conn)

            for version, sql_text in migrations:
                chk = _checksum(sql_text)
                if version in existing:
                    if existing[version] != chk:
                        raise MigrationError(
                            f"checksum mismatch for applied migration {version}: "
                            f"expected {existing[version]}, found {chk}"
                        )
                    continue

                try:
                    with conn.transaction():
                        with conn.cursor() as cur:
                            cur.execute(sql_text)
                        _record_migration(conn, version, version, chk)
                except MigrationError:
                    raise
                except Exception as exc:
                    raise MigrationError(
                        f"failed to apply migration {version}: {exc}"
                    ) from exc

                applied_count += 1
                logger.info("migration applied", extra={"version": version})

    except MigrationError:
        raise
    except Exception as exc:
        raise MigrationError(
            f"cannot connect to PostgreSQL for migrations: {exc}"
        ) from exc

    return applied_count


def _dsn_from_env() -> str:
    """Build a libpq DSN from environment variables."""
    parts = [
        f"host={os.environ.get('PGHOST', 'localhost')}",
        f"port={os.environ.get('PGPORT', '5432')}",
        f"dbname={os.environ.get('PGDATABASE', 'riskforge')}",
        f"user={os.environ.get('PGUSER', 'postgres')}",
    ]
    password = os.environ.get("PGPASSWORD", "")
    if password:
        parts.append("password=***")  # mask password in DSN
    return " ".join(parts)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point: ``python -m riskforge.persistence.postgres.migrate``."""
    try:
        count = run_migrations()
        logger.info("migrations applied", extra={"count": count})
    except MigrationError as exc:
        logger.error("migration failed", extra={"error": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
