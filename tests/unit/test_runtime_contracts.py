from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskforge.runtime.contracts import (
    ComponentReadiness,
    LifecycleState,
    ReadinessState,
    RuntimeAssembly,
    RuntimeSettings,
    compose_readiness,
)


class Component:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def readiness(self) -> ComponentReadiness:
        return ComponentReadiness(self.name, True)


def test_runtime_settings_are_strict_portable_and_secret_free() -> None:
    settings = RuntimeSettings(
        model_path=Path("models") / "riskforge.onnx",
        manifest_path=Path("models") / "manifest.json",
    )
    assert settings.max_batch_size == 32
    assert "password" not in settings.model_dump()
    assert "database_url" not in settings.model_dump()
    with pytest.raises(ValidationError):
        RuntimeSettings(
            model_path="model.onnx",
            manifest_path="manifest.json",
            max_batch_size=33,
        )
    with pytest.raises(ValidationError):
        RuntimeSettings(
            model_path="model.onnx",
            manifest_path="manifest.json",
            linux_only_setting=True,
        )


def test_readiness_requires_ready_lifecycle_and_every_component() -> None:
    checked_at = datetime.now(UTC)
    components = (ComponentReadiness("model", True), ComponentReadiness("storage", False))
    status = compose_readiness(LifecycleState.READY, components, checked_at=checked_at)
    assert status.readiness is ReadinessState.NOT_READY
    assert status.components == components

    ready = compose_readiness(
        LifecycleState.READY,
        (ComponentReadiness("model", True), ComponentReadiness("storage", True)),
        checked_at=checked_at,
    )
    assert ready.readiness is ReadinessState.READY


def test_readiness_preserves_declared_order_and_rejects_duplicate_names() -> None:
    checked_at = datetime.now(UTC)
    status = compose_readiness(
        LifecycleState.DEGRADED,
        (ComponentReadiness("second", True), ComponentReadiness("first", True)),
        checked_at=checked_at,
    )
    assert [item.name for item in status.components] == ["second", "first"]
    with pytest.raises(ValueError, match="unique"):
        compose_readiness(
            LifecycleState.READY,
            (ComponentReadiness("same", True), ComponentReadiness("same", True)),
            checked_at=checked_at,
        )


def test_runtime_assembly_validates_component_identity() -> None:
    assembly = RuntimeAssembly(application=object(), components=(Component("model"),))
    assert assembly.components[0].name == "model"
    with pytest.raises(ValueError, match="unique"):
        RuntimeAssembly(
            application=object(), components=(Component("duplicate"), Component("duplicate"))
        )


def test_readiness_details_are_bounded() -> None:
    with pytest.raises(ValueError, match="too long"):
        ComponentReadiness("model", False, "x" * 201)
