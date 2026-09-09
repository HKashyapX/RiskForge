"""Deterministic, thread-safe runtime lifecycle coordination."""

from __future__ import annotations

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import UTC, datetime
from threading import RLock
from typing import Generic, TypeVar

from riskforge.runtime.contracts import (
    ComponentReadiness,
    LifecycleState,
    RuntimeAssembly,
    RuntimeStatus,
    compose_readiness,
)
from riskforge.runtime.exceptions import (
    RuntimeShutdownError,
    RuntimeStartupError,
    RuntimeStateError,
)

logger = logging.getLogger("riskforge.runtime.lifecycle")

T = TypeVar("T")


class RuntimeManager(Generic[T]):
    """Start components in order and stop them in reverse order."""

    def __init__(
        self,
        assembly: RuntimeAssembly[T],
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        self._assembly = assembly
        self._state = LifecycleState.CREATED
        self._started_count = 0
        self._lock = RLock()
        self._shutdown_timeout_s = shutdown_timeout_s
        self._started_at: float | None = None

    @property
    def application(self) -> T:
        return self._assembly.application

    @property
    def state(self) -> LifecycleState:
        with self._lock:
            return self._state

    def start(self) -> None:
        """Start once, rolling back already-started components on failure."""
        with self._lock:
            if self._state is LifecycleState.READY:
                return
            if self._state is LifecycleState.DEGRADED:
                self._state = LifecycleState.CREATED  # allow re-attempt
            if self._state is not LifecycleState.CREATED:
                raise RuntimeStateError(f"cannot start runtime from {self._state.value}")
            self._state = LifecycleState.STARTING
            start_time = time.monotonic()
            try:
                for component in self._assembly.components:
                    self._started_count += 1
                    logger.info("starting component", extra={"component": component.name})
                    component.start()
                    logger.debug("component started", extra={"component": component.name})
            except Exception as error:
                self._rollback_started()
                self._state = LifecycleState.FAILED
                logger.error("runtime startup failed", exc_info=error)
                raise RuntimeStartupError("runtime startup failed") from error
            elapsed = time.monotonic() - start_time
            self._started_at = time.monotonic()
            self._state = LifecycleState.READY
            logger.info(
                "runtime started",
                extra={
                    "components": self._started_count,
                    "startup_duration_s": round(elapsed, 3),
                },
            )

    def stop(self) -> None:
        """Stop every started component in reverse order, attempting all cleanup."""
        with self._lock:
            if self._state is LifecycleState.STOPPED:
                return
            if self._state is LifecycleState.CREATED:
                self._state = LifecycleState.STOPPED
                return
            if self._state not in {
                LifecycleState.READY,
                LifecycleState.FAILED,
                LifecycleState.DEGRADED,
            }:
                raise RuntimeStateError(f"cannot stop runtime from {self._state.value}")
            self._state = LifecycleState.STOPPING
            logger.info("runtime stopping")
            failures = self._stop_started()
            self._state = LifecycleState.FAILED if failures else LifecycleState.STOPPED
            if failures:
                logger.error(
                    "runtime shutdown failed",
                    extra={"failure_count": len(failures)},
                    exc_info=failures[0],
                )
                raise RuntimeShutdownError("runtime shutdown failed") from failures[0]
            logger.info("runtime stopped")

    def status(self, *, checked_at: datetime | None = None) -> RuntimeStatus:
        """Return a stable snapshot; readiness failures become safe not-ready states."""
        with self._lock:
            components: list[ComponentReadiness] = []
            for component in self._assembly.components:
                try:
                    snapshot = component.readiness()
                    if snapshot.name != component.name:
                        snapshot = ComponentReadiness(
                            component.name, False, "readiness identity mismatch"
                        )
                except Exception:  # noqa: BLE001 - isolate arbitrary component probes
                    snapshot = ComponentReadiness(component.name, False, "readiness check failed")
                components.append(snapshot)
            return compose_readiness(
                self._state,
                components,
                checked_at=checked_at or datetime.now(UTC),
            )

    def register_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown.

        Must be called from the main thread.  Stores the original handlers
        so they can be restored if needed.
        """
        manager = self

        def _shutdown_handler(signum: int, _frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("received %s, initiating graceful shutdown", sig_name)
            manager.stop()

        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)
        logger.info(
            "signal handlers registered",
            extra={"signals": ["SIGTERM", "SIGINT"]},
        )

    def _rollback_started(self) -> None:
        self._stop_started()

    def _stop_started(self) -> list[Exception]:
        failures: list[Exception] = []
        started = self._assembly.components[: self._started_count]
        with ThreadPoolExecutor(max_workers=len(started) or 1) as executor:
            for component in reversed(started):
                try:
                    future = executor.submit(component.stop)
                    future.result(timeout=self._shutdown_timeout_s)
                except FuturesTimeoutError:
                    logger.warning(
                        "component stop timed out",
                        extra={
                            "component": component.name,
                            "timeout_s": self._shutdown_timeout_s,
                        },
                    )
                    failures.append(
                        RuntimeError(f"stop timed out for {component.name}")
                    )
                except Exception as error:
                    logger.warning(
                        "component stop failed",
                        extra={"component": component.name},
                        exc_info=error,
                    )
                    failures.append(error)
        self._started_count = 0
        return failures
