"""Tests for resilience patterns: circuit breaker, retry, exception taxonomy, lifecycle."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from riskforge.persistence.exceptions import (
    PersistenceConnectionError,
    PersistenceError,
    PersistenceTimeoutError,
)
from riskforge.persistence.postgres.connection import (
    _CIRCUIT_BREAKER_THRESHOLD,
    PoolReadiness,
    PostgresConfig,
    PostgresConnectionPool,
)
from riskforge.runtime.contracts import (
    ComponentReadiness,
    LifecycleState,
    RuntimeAssembly,
)
from riskforge.runtime.exceptions import RuntimeShutdownError
from riskforge.runtime.lifecycle import RuntimeManager
from riskforge.runtime.persistence_adapter import PostgresPoolLifecycle

# ---------------------------------------------------------------------------
# Persistence exception taxonomy
# ---------------------------------------------------------------------------


class TestPersistenceExceptions:
    def test_connection_error_is_persistence_error(self) -> None:
        assert issubclass(PersistenceConnectionError, PersistenceError)

    def test_timeout_error_is_persistence_error(self) -> None:
        assert issubclass(PersistenceTimeoutError, PersistenceError)

    def test_connection_error_default_message(self) -> None:
        exc = PersistenceConnectionError()
        assert "unavailable" in str(exc)

    def test_timeout_error_default_message(self) -> None:
        exc = PersistenceTimeoutError()
        assert "timed out" in str(exc)

    def test_connection_error_custom_message(self) -> None:
        exc = PersistenceConnectionError("custom msg")
        assert str(exc) == "custom msg"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initial_state_closed(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        assert pool._is_circuit_open() is False

    def test_circuit_opens_after_threshold(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = _CIRCUIT_BREAKER_THRESHOLD
        pool._circuit_open_until = time.monotonic() + 100  # far in the future
        assert pool._is_circuit_open() is True

    def test_circuit_half_open_after_cooldown(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = _CIRCUIT_BREAKER_THRESHOLD
        pool._circuit_open_until = time.monotonic() - 1  # in the past
        assert pool._is_circuit_open() is False

    def test_record_success_resets_counter(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = 3
        pool._circuit_open_until = 0.0
        pool._record_success()
        assert pool._consecutive_failures == 0

    def test_record_failure_increments_counter(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        pool._record_failure()
        assert pool._consecutive_failures == 1

    def test_record_failure_opens_circuit_at_threshold(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = _CIRCUIT_BREAKER_THRESHOLD - 1
        pool._circuit_open_until = 0.0
        pool._record_failure()
        assert pool._consecutive_failures == _CIRCUIT_BREAKER_THRESHOLD
        assert pool._circuit_open_until > time.monotonic()

    def test_getconn_raises_when_circuit_open(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = _CIRCUIT_BREAKER_THRESHOLD
        pool._circuit_open_until = time.monotonic() + 100
        with pytest.raises(PersistenceConnectionError, match="circuit breaker"):
            pool.getconn()

    def test_close_resets_circuit_breaker(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._consecutive_failures = 10
        pool._circuit_open_until = time.monotonic() + 100
        pool._pool = MagicMock()
        pool.close()
        assert pool._consecutive_failures == 0
        assert pool._circuit_open_until == 0.0


# ---------------------------------------------------------------------------
# PostgresConfig DSN masking
# ---------------------------------------------------------------------------


class TestPostgresConfigDsn:
    def test_dsn_masks_password(self) -> None:
        config = PostgresConfig(
            host="localhost",
            port=5432,
            dbname="riskforge",
            user="postgres",
            password="supersecret",
            connect_timeout=10,
            min_pool_size=1,
            max_pool_size=5,
        )
        dsn = config.dsn()
        assert "supersecret" not in dsn
        assert "***" in dsn

    def test_dsn_full_includes_password(self) -> None:
        config = PostgresConfig(
            host="localhost",
            port=5432,
            dbname="riskforge",
            user="postgres",
            password="supersecret",
            connect_timeout=10,
            min_pool_size=1,
            max_pool_size=5,
        )
        dsn = config._dsn_full()
        assert "supersecret" in dsn


# ---------------------------------------------------------------------------
# Pool lifecycle: open / readiness / close / timeout translation
# ---------------------------------------------------------------------------


class TestPoolLifecycle:
    def test_stable_component_name(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        assert pool.name == "postgres_pool"

    def test_open_creates_pool_and_waits(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        pool._get_pool = lambda: fake_pool  # type: ignore[method-assign]
        pool.open()
        fake_pool.wait.assert_called_once_with(timeout=30.0)

    def test_open_timeout_raises_persistence_timeout(self) -> None:
        from psycopg_pool import PoolTimeout

        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        fake_pool.wait.side_effect = PoolTimeout("pool timeout")
        pool._get_pool = lambda: fake_pool  # type: ignore[method-assign]
        with pytest.raises(PersistenceTimeoutError):
            pool.open()

    def test_open_generic_failure_raises_connection_error(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        fake_pool.wait.side_effect = RuntimeError("boom")
        pool._get_pool = lambda: fake_pool  # type: ignore[method-assign]
        with pytest.raises(PersistenceConnectionError):
            pool.open()

    def test_open_records_failure_on_error(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        fake_pool.wait.side_effect = RuntimeError("boom")
        pool._get_pool = lambda: fake_pool  # type: ignore[method-assign]
        with pytest.raises(PersistenceConnectionError):
            pool.open()
        assert pool._consecutive_failures == 1

    def test_readiness_not_ready_before_open(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        snapshot = pool.readiness()
        assert isinstance(snapshot, PoolReadiness)
        assert snapshot.ready is False
        assert snapshot.detail == "pool not created"

    def test_readiness_not_ready_when_circuit_open(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._consecutive_failures = _CIRCUIT_BREAKER_THRESHOLD
        pool._circuit_open_until = time.monotonic() + 100
        snapshot = pool.readiness()
        assert snapshot.ready is False
        assert snapshot.detail == "circuit breaker open"

    def test_readiness_ready_when_connection_works(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        snapshot = pool.readiness()
        assert snapshot.ready is True
        assert snapshot.detail == "pool ready"
        pool._pool.putconn.assert_called_once()

    def test_readiness_not_ready_on_acquisition_timeout(self) -> None:
        from psycopg_pool import PoolTimeout

        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._pool.getconn.side_effect = PoolTimeout("pool timeout")
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        snapshot = pool.readiness()
        assert snapshot.ready is False
        assert snapshot.detail == "connection acquisition timed out"

    def test_readiness_probe_is_read_only(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._pool.getconn.side_effect = RuntimeError("boom")
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        pool.readiness()
        assert pool._consecutive_failures == 0

    def test_close_is_idempotent(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        pool._pool = fake_pool
        pool.close()
        pool.close()
        fake_pool.close.assert_called_once()
        assert pool._pool is None

    def test_getconn_translates_timeout_after_retries(self) -> None:
        from psycopg_pool import PoolTimeout

        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._pool.getconn.side_effect = PoolTimeout("pool timeout")
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        with (
            patch("riskforge.persistence.postgres.connection.time.sleep"),
            pytest.raises(PersistenceTimeoutError),
        ):
            pool.getconn()

    def test_getconn_generic_error_after_retries_is_connection_error(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._pool.getconn.side_effect = RuntimeError("refused")
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        with (
            patch("riskforge.persistence.postgres.connection.time.sleep"),
            pytest.raises(PersistenceConnectionError),
        ):
            pool.getconn()


# ---------------------------------------------------------------------------
# Pool-to-runtime lifecycle adapter
# ---------------------------------------------------------------------------


class TestPostgresPoolLifecycleAdapter:
    def test_satisfies_lifecycle_component_protocol(self) -> None:
        from riskforge.runtime.contracts import LifecycleComponent

        adapter = PostgresPoolLifecycle(pool=PostgresConnectionPool(config=PostgresConfig()))
        assert isinstance(adapter, LifecycleComponent)

    def test_name_delegates_to_pool(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        adapter = PostgresPoolLifecycle(pool=pool)
        assert adapter.name == "postgres_pool"

    def test_start_opens_pool(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        pool._get_pool = lambda: fake_pool  # type: ignore[method-assign]
        adapter = PostgresPoolLifecycle(pool=pool)
        adapter.start()
        fake_pool.wait.assert_called_once_with(timeout=30.0)

    def test_stop_closes_pool(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        pool._pool = fake_pool
        adapter = PostgresPoolLifecycle(pool=pool)
        adapter.stop()
        fake_pool.close.assert_called_once()

    def test_stop_is_idempotent(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        pool._pool = fake_pool
        adapter = PostgresPoolLifecycle(pool=pool)
        adapter.stop()
        adapter.stop()
        fake_pool.close.assert_called_once()

    def test_readiness_translates_pool_snapshot(self) -> None:
        pool = PostgresConnectionPool.__new__(PostgresConnectionPool)
        pool._config = PostgresConfig()
        pool._pool = MagicMock()
        pool._consecutive_failures = 0
        pool._circuit_open_until = 0.0
        adapter = PostgresPoolLifecycle(pool=pool)
        snapshot = adapter.readiness()
        assert isinstance(snapshot, ComponentReadiness)
        assert snapshot.name == "postgres_pool"
        assert snapshot.ready is True
        assert snapshot.detail == "pool ready"

    def test_readiness_translates_not_ready_snapshot(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        adapter = PostgresPoolLifecycle(pool=pool)
        snapshot = adapter.readiness()
        assert snapshot.name == "postgres_pool"
        assert snapshot.ready is False
        assert snapshot.detail == "pool not created"

    def test_adapter_composes_with_runtime_manager(self) -> None:
        pool = PostgresConnectionPool(config=PostgresConfig())
        fake_pool = MagicMock()
        pool._pool = fake_pool
        adapter = PostgresPoolLifecycle(pool=pool)
        assembly = RuntimeAssembly(application=_StubApp(), components=(adapter,))
        manager = RuntimeManager(assembly)

        manager.start()
        assert manager.state == LifecycleState.READY
        status = manager.status()
        assert status.components[0].name == "postgres_pool"
        assert status.components[0].ready is True

        manager.stop()
        assert manager.state == LifecycleState.STOPPED
        fake_pool.close.assert_called_once()


# ---------------------------------------------------------------------------
# Runtime lifecycle
# ---------------------------------------------------------------------------


class _StubComponent:
    def __init__(self, name: str, *, fail_start: bool = False, fail_stop: bool = False) -> None:
        self.name = name
        self._started = False
        self._stopped = False
        self._fail_start = fail_start
        self._fail_stop = fail_stop

    def start(self) -> None:
        if self._fail_start:
            raise RuntimeError(f"{self.name} start failed")
        self._started = True

    def stop(self) -> None:
        if self._fail_stop:
            raise RuntimeError(f"{self.name} stop failed")
        self._stopped = True

    def readiness(self) -> ComponentReadiness:
        return ComponentReadiness(self.name, self._started, "ok")


class _StubApp:
    pass


class TestRuntimeManager:
    def test_start_stop_lifecycle(self) -> None:
        comp = _StubComponent("test")
        assembly = RuntimeAssembly(application=_StubApp(), components=[comp])
        manager = RuntimeManager(assembly)
        assert manager.state == LifecycleState.CREATED

        manager.start()
        assert manager.state == LifecycleState.READY
        assert comp._started is True

        manager.stop()
        assert manager.state == LifecycleState.STOPPED
        assert comp._stopped is True

    def test_start_from_degraded_allows_restart(self) -> None:
        comp = _StubComponent("test")
        assembly = RuntimeAssembly(application=_StubApp(), components=[comp])
        manager = RuntimeManager(assembly)
        manager._state = LifecycleState.DEGRADED

        manager.start()
        assert manager.state == LifecycleState.READY

    def test_stop_from_degraded_succeeds(self) -> None:
        comp = _StubComponent("test")
        assembly = RuntimeAssembly(application=_StubApp(), components=[comp])
        manager = RuntimeManager(assembly)
        manager._state = LifecycleState.DEGRADED
        manager._started_count = 1

        manager.stop()
        assert manager.state == LifecycleState.STOPPED

    def test_stop_timeout(self) -> None:
        class _SlowComponent:
            name = "slow"

            def start(self) -> None:
                pass

            def stop(self) -> None:
                time.sleep(10)  # will timeout

            def readiness(self) -> ComponentReadiness:
                return ComponentReadiness(self.name, True, "ok")

        assembly = RuntimeAssembly(application=_StubApp(), components=[_SlowComponent()])
        manager = RuntimeManager(assembly, shutdown_timeout_s=0.1)
        manager.start()
        assert manager.state == LifecycleState.READY

        with pytest.raises(RuntimeShutdownError):
            manager.stop()

    def test_status_snapshot(self) -> None:
        comp = _StubComponent("test")
        assembly = RuntimeAssembly(application=_StubApp(), components=[comp])
        manager = RuntimeManager(assembly)
        manager.start()
        status = manager.status()
        assert status.lifecycle == LifecycleState.READY
        assert len(status.components) == 1
        assert status.components[0].name == "test"
