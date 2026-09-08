from datetime import UTC, datetime

import pytest

from riskforge.runtime import (
    ComponentReadiness,
    LifecycleState,
    ReadinessState,
    RuntimeAssembly,
    RuntimeManager,
    RuntimeReadinessProvider,
    RuntimeShutdownError,
    RuntimeStartupError,
    RuntimeStateError,
)


class Component:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
        readiness_error: Exception | None = None,
        ready: bool = True,
    ) -> None:
        self._name = name
        self.events = events
        self.start_error = start_error
        self.stop_error = stop_error
        self.readiness_error = readiness_error
        self.ready = ready

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")
        if self.stop_error is not None:
            raise self.stop_error

    def readiness(self) -> ComponentReadiness:
        if self.readiness_error is not None:
            raise self.readiness_error
        return ComponentReadiness(self.name, self.ready)


def _manager(*components: Component) -> RuntimeManager[str]:
    return RuntimeManager(RuntimeAssembly(application="application", components=components))


def test_lifecycle_starts_in_order_and_stops_in_reverse_order() -> None:
    events: list[str] = []
    manager = _manager(Component("model", events), Component("queue", events))

    manager.start()
    manager.start()
    assert manager.state is LifecycleState.READY
    assert manager.application == "application"

    manager.stop()
    manager.stop()
    assert manager.state is LifecycleState.STOPPED
    assert events == ["start:model", "start:queue", "stop:queue", "stop:model"]


def test_startup_failure_rolls_back_only_started_components() -> None:
    events: list[str] = []
    manager = _manager(
        Component("model", events),
        Component("queue", events, start_error=RuntimeError("private")),
        Component("server", events),
    )

    with pytest.raises(RuntimeStartupError) as caught:
        manager.start()
    assert manager.state is LifecycleState.FAILED
    assert events == ["start:model", "start:queue", "stop:queue", "stop:model"]
    assert isinstance(caught.value.__cause__, RuntimeError)
    with pytest.raises(RuntimeStateError):
        manager.start()


def test_shutdown_attempts_every_component_after_failure() -> None:
    events: list[str] = []
    manager = _manager(
        Component("model", events),
        Component("queue", events, stop_error=RuntimeError("private")),
    )
    manager.start()

    with pytest.raises(RuntimeShutdownError):
        manager.stop()
    assert manager.state is LifecycleState.FAILED
    assert events[-2:] == ["stop:queue", "stop:model"]


def test_status_contains_safe_failure_and_deterministic_order() -> None:
    events: list[str] = []
    manager = _manager(
        Component("model", events),
        Component(
            "queue",
            events,
            readiness_error=RuntimeError("Sensitive incident narrative"),
        ),
    )
    manager.start()
    checked_at = datetime(2026, 1, 1, tzinfo=UTC)

    status = manager.status(checked_at=checked_at)
    assert status.readiness is ReadinessState.NOT_READY
    assert [component.name for component in status.components] == ["model", "queue"]
    assert status.components[1].detail == "readiness check failed"
    assert "Sensitive incident narrative" not in repr(status)


def test_api_readiness_adapter_exposes_only_transport_safe_state() -> None:
    events: list[str] = []
    manager = _manager(Component("model", events))
    manager.start()

    snapshot = RuntimeReadinessProvider(manager).snapshot()
    assert snapshot.ready is True
    assert snapshot.state == "ready"
    assert snapshot.components == ("model",)
