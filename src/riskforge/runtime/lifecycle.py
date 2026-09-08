"""Deterministic, thread-safe runtime lifecycle coordination."""

from __future__ import annotations

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

T = TypeVar("T")


class RuntimeManager(Generic[T]):
    """Start components in order and stop them in reverse order."""

    def __init__(self, assembly: RuntimeAssembly[T]) -> None:
        self._assembly = assembly
        self._state = LifecycleState.CREATED
        self._started_count = 0
        self._lock = RLock()

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
            if self._state is not LifecycleState.CREATED:
                raise RuntimeStateError(f"cannot start runtime from {self._state.value}")
            self._state = LifecycleState.STARTING
            try:
                for component in self._assembly.components:
                    self._started_count += 1
                    component.start()
            except Exception as error:
                self._rollback_started()
                self._state = LifecycleState.FAILED
                raise RuntimeStartupError("runtime startup failed") from error
            self._state = LifecycleState.READY

    def stop(self) -> None:
        """Stop every started component in reverse order, attempting all cleanup."""
        with self._lock:
            if self._state is LifecycleState.STOPPED:
                return
            if self._state is LifecycleState.CREATED:
                self._state = LifecycleState.STOPPED
                return
            if self._state not in {LifecycleState.READY, LifecycleState.FAILED}:
                raise RuntimeStateError(f"cannot stop runtime from {self._state.value}")
            self._state = LifecycleState.STOPPING
            failures = self._stop_started()
            self._state = LifecycleState.FAILED if failures else LifecycleState.STOPPED
            if failures:
                raise RuntimeShutdownError("runtime shutdown failed") from failures[0]

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

    def _rollback_started(self) -> None:
        self._stop_started()

    def _stop_started(self) -> list[Exception]:
        failures: list[Exception] = []
        started = self._assembly.components[: self._started_count]
        for component in reversed(started):
            try:
                component.stop()
            except Exception as error:  # noqa: BLE001 - continue best-effort component cleanup
                failures.append(error)
        self._started_count = 0
        return failures
