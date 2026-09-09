"""Migration smoke tests.

Verifies that the versioned migration system correctly initializes an
empty PostgreSQL database and is safe to run repeatedly.

Requires a running PostgreSQL instance.  Tests are skipped automatically
when the database is unreachable.

Strategy: uses a single test database and TRUNCATE/drops the tracking
table between tests for isolation, instead of creating/dropping databases
(which has flaky CREATE DATABASE semantics with psycopg3).
"""

from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path

import psycopg
import pytest

_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "riskforge"
    / "persistence"
    / "postgres"
    / "migrations"
)


def _pg_available() -> bool:
    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "5432"))
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _dsn() -> str:
    """Build DSN from environment variables."""
    parts = [
        f"host={os.environ.get('PGHOST', 'localhost')}",
        f"port={os.environ.get('PGPORT', '5432')}",
        f"dbname={os.environ.get('PGDATABASE', 'riskforge_test')}",
        f"user={os.environ.get('PGUSER', 'postgres')}",
    ]
    password = os.environ.get("PGPASSWORD", "")
    if password:
        parts.append(f"password={password}")
    return " ".join(parts)


def _reset_schema(dsn: str) -> None:
    """Drop all tables including schema_migrations to get a truly empty database."""
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS incident_results CASCADE")
            cur.execute("DROP TABLE IF EXISTS review_decisions CASCADE")
            cur.execute("DROP TABLE IF EXISTS audit_events CASCADE")
            cur.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")


def _table_names(dsn: str) -> set[str]:
    """Return the set of user table names in the database."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
        return {row[0] for row in cur.fetchall()}


def _index_names(dsn: str) -> set[str]:
    """Return the set of user index names in the database."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
        )
        return {row[0] for row in cur.fetchall()}


def _tracking_rows(dsn: str) -> dict[str, str]:
    """Return {version: checksum} from the schema_migrations table."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations")
        return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _pg_available(), reason="PostgreSQL not available")
class TestMigrationSmoke:
    """Core migration behaviour on a fresh database."""

    def test_empty_database_to_current_schema(self) -> None:
        """Migrations applied to an empty database produce all required tables."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import run_migrations

            applied = run_migrations(dsn=dsn)
            assert applied >= 1
            tables = _table_names(dsn)
            assert "incident_results" in tables
            assert "review_decisions" in tables
            assert "audit_events" in tables
        finally:
            _reset_schema(dsn)

    def test_repeated_migration_execution_is_idempotent(self) -> None:
        """Running migrations twice applies only once."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import run_migrations

            first = run_migrations(dsn=dsn)
            assert first >= 1
            second = run_migrations(dsn=dsn)
            assert second == 0
        finally:
            _reset_schema(dsn)

    def test_required_indexes_exist(self) -> None:
        """All expected indexes are created by the migration."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import run_migrations

            run_migrations(dsn=dsn)
            indexes = _index_names(dsn)
            assert "idx_incident_results_timestamp_log_id" in indexes
            assert "idx_incident_results_asset_id" in indexes
            assert "idx_incident_results_asset_type" in indexes
            assert "idx_incident_results_routing" in indexes
            assert "idx_incident_results_score" in indexes
            assert "idx_review_decisions_incident" in indexes
            assert "idx_audit_events_incident" in indexes
        finally:
            _reset_schema(dsn)

    def test_tracking_table_records_migration(self) -> None:
        """schema_migrations contains a row for each applied migration."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import run_migrations

            run_migrations(dsn=dsn)
            rows = _tracking_rows(dsn)
            assert "001_initial" in rows
            # Verify checksum matches the actual file content
            migration_file = _MIGRATIONS_DIR / "001_initial.sql"
            expected_chk = hashlib.sha256(
                migration_file.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            assert rows["001_initial"] == expected_chk
        finally:
            _reset_schema(dsn)

    def test_tracking_table_checksum_mismatch_detected(self) -> None:
        """If a migration file is modified after application, detect the mismatch."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import (
                MigrationError,
                run_migrations,
            )

            run_migrations(dsn=dsn)
            # Tamper with the checksum in the tracking table
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE schema_migrations SET checksum = 'tampered' "
                        "WHERE version = '001_initial'"
                    )
                conn.commit()
            # Re-run should detect the mismatch
            with pytest.raises(MigrationError, match="checksum mismatch"):
                run_migrations(dsn=dsn)
        finally:
            _reset_schema(dsn)

    def test_migration_creates_tracking_table(self) -> None:
        """The schema_migrations tracking table is created on first run."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import run_migrations

            run_migrations(dsn=dsn)
            tables = _table_names(dsn)
            assert "schema_migrations" in tables
        finally:
            _reset_schema(dsn)

    def test_required_primary_key_constraints(self) -> None:
        """Primary key columns reject duplicates."""
        dsn = _dsn()
        _reset_schema(dsn)
        try:
            from riskforge.persistence.postgres.migrate import run_migrations

            run_migrations(dsn=dsn)
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    # Insert a row
                    cur.execute(
                        "INSERT INTO incident_results "
                        "(log_id, timestamp, asset_id, asset_type, routing, "
                        "calibrated_sif_p_score, record_json) "
                        "VALUES ('PK_TEST', now(), 'A', 'B', 'C', 0.5, '{}')"
                    )
                    # Duplicate primary key should fail
                    with pytest.raises(psycopg.errors.UniqueViolation):
                        cur.execute(
                            "INSERT INTO incident_results "
                            "(log_id, timestamp, asset_id, asset_type, routing, "
                            "calibrated_sif_p_score, record_json) "
                            "VALUES ('PK_TEST', now(), 'A', 'B', 'C', 0.5, '{}')"
                        )
                conn.rollback()
        finally:
            _reset_schema(dsn)
