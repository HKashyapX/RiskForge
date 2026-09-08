"""Adapter from runtime status to the API-owned readiness port."""

from typing import Generic, TypeVar

from riskforge.api.dependencies import ReadinessSnapshot
from riskforge.runtime.lifecycle import RuntimeManager

T = TypeVar("T")


class RuntimeReadinessProvider(Generic[T]):
    def __init__(self, manager: RuntimeManager[T]) -> None:
        self._manager = manager

    def snapshot(self) -> ReadinessSnapshot:
        status = self._manager.status()
        return ReadinessSnapshot(
            ready=status.readiness.value == "ready",
            state=status.readiness.value,
            checked_at=status.checked_at,
            components=tuple(component.name for component in status.components),
        )
