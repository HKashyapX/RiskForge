"""Lifecycle adapter binding the PostgreSQL pool to the runtime lifecycle."""

from __future__ import annotations

from riskforge.persistence.postgres.connection import (
    PoolReadiness,
    PostgresConnectionPool,
)
from riskforge.runtime.contracts import ComponentReadiness


class PostgresPoolLifecycle:
    """Adapt :class:`PostgresConnectionPool` to the runtime component protocol.

    Runtime composition (ADR-0005) owns lifecycle ordering and readiness
    aggregation, while persistence owns pool behavior.  This adapter keeps
    the two subsystems decoupled: the runtime sees a standard
    ``LifecycleComponent`` and never imports pool internals.

    - ``start`` performs the fail-fast pool open: an unreachable database
      blocks startup instead of failing on the first request.
    - ``stop`` is the idempotent pool shutdown.
    - ``readiness`` translates the persistence-owned snapshot into the
      runtime-owned contract without mutating circuit-breaker state.
    """

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    @property
    def name(self) -> str:
        """Return the stable component name from the pool."""
        return self._pool.name

    def start(self) -> None:
        """Open the pool and wait until its minimum connections are ready."""
        self._pool.open()

    def stop(self) -> None:
        """Close the pool idempotently."""
        self._pool.close()

    def readiness(self) -> ComponentReadiness:
        """Translate the pool readiness snapshot into the runtime contract."""
        snapshot: PoolReadiness = self._pool.readiness()
        return ComponentReadiness(snapshot.name, snapshot.ready, snapshot.detail)
