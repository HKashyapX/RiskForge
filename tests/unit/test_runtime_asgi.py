from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from riskforge.runtime.asgi import create_managed_app
from riskforge.runtime.contracts import (
    ComponentReadiness,
    LifecycleState,
    RuntimeAssembly,
    RuntimeSettings,
)
from riskforge.runtime.exceptions import RuntimeStartupError


class Application:
    pass


class Component:
    def __init__(self, name: str, events: list[str], *, fail_start: bool = False) -> None:
        self._name = name
        self._events = events
        self._ready = False
        self._fail_start = fail_start

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self._events.append(f"start:{self.name}")
        if self._fail_start:
            raise RuntimeError("private startup detail")
        self._ready = True

    def stop(self) -> None:
        self._events.append(f"stop:{self.name}")
        self._ready = False

    def readiness(self) -> ComponentReadiness:
        return ComponentReadiness(self.name, self._ready)


class Composer:
    def __init__(self, components: tuple[Component, ...]) -> None:
        self.components = components
        self.settings: RuntimeSettings | None = None

    def compose(self, settings: RuntimeSettings) -> RuntimeAssembly[Application]:
        self.settings = settings
        return RuntimeAssembly(Application(), self.components)


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        model_path=Path("model.onnx"),
        manifest_path=Path("manifest.json"),
        max_request_bytes=2_048,
        max_batch_size=4,
        request_timeout_seconds=2.0,
        shutdown_timeout_seconds=1.0,
    )


def test_managed_app_starts_checks_readiness_and_stops_in_reverse_order() -> None:
    events: list[str] = []
    model = Component("model", events)
    storage = Component("storage", events)
    composer = Composer((model, storage))
    app = create_managed_app(
        _settings(), composer, enable_metrics=False, cors_origins=[]
    )
    manager = app.state.runtime_manager

    assert manager.state is LifecycleState.CREATED
    with TestClient(app) as client:
        assert manager.state is LifecycleState.READY
        response = client.get(
            "/ready", headers={"X-Correlation-ID": "runtime-probe"}
        )
        assert response.status_code == 200
        assert response.json()["components"] == ["model", "storage"]

    assert manager.state is LifecycleState.STOPPED
    assert events == [
        "start:model",
        "start:storage",
        "stop:storage",
        "stop:model",
    ]
    assert composer.settings == _settings()


def test_managed_app_rolls_back_when_startup_fails() -> None:
    events: list[str] = []
    first = Component("storage", events)
    failing = Component("model", events, fail_start=True)
    app = create_managed_app(
        _settings(), Composer((first, failing)), enable_metrics=False, cors_origins=[]
    )

    with pytest.raises(RuntimeStartupError, match="runtime startup failed"), TestClient(app):
        pass

    assert app.state.runtime_manager.state is LifecycleState.FAILED
    assert events == ["start:storage", "start:model", "stop:model", "stop:storage"]
