"""Portable settings, lifecycle, and dependency-composition contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RuntimeEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LifecycleState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ReadinessState(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"


class RuntimeSettings(BaseModel):
    """Validated non-secret settings shared by runtime implementations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT
    api_host: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    api_port: int = Field(default=8000, ge=1, le=65_535)
    model_path: Path
    manifest_path: Path
    max_request_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)
    max_batch_size: int = Field(default=32, ge=1, le=32)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    shutdown_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    inference_queue_size: int = Field(default=256, ge=1, le=10_000)
    inference_queue_delay_ms: float = Field(default=5.0, ge=0.0, le=1_000.0)


@dataclass(frozen=True)
class ComponentReadiness:
    name: str
    ready: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("component readiness name must not be empty")
        if self.detail is not None and len(self.detail) > 200:
            raise ValueError("component readiness detail is too long")


@dataclass(frozen=True)
class RuntimeStatus:
    lifecycle: LifecycleState
    readiness: ReadinessState
    checked_at: datetime
    components: tuple[ComponentReadiness, ...]


@runtime_checkable
class LifecycleComponent(Protocol):
    @property
    def name(self) -> str:
        """Return the stable component name."""

    def start(self) -> None:
        """Start and validate the component."""

    def stop(self) -> None:
        """Stop the component idempotently."""

    def readiness(self) -> ComponentReadiness:
        """Return a transport-neutral readiness snapshot."""


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class RuntimeAssembly(Generic[T]):
    """Constructed application dependency plus ordered lifecycle components."""

    application: T
    components: tuple[LifecycleComponent, ...]

    def __post_init__(self) -> None:
        names = [component.name for component in self.components]
        if any(not name.strip() for name in names):
            raise ValueError("runtime component names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("runtime component names must be unique")


@runtime_checkable
class DependencyComposer(Protocol, Generic[T_co]):
    def compose(self, settings: RuntimeSettings) -> RuntimeAssembly[T_co]:
        """Build dependencies without starting their lifecycle."""


def compose_readiness(
    lifecycle: LifecycleState,
    components: Sequence[ComponentReadiness],
    *,
    checked_at: datetime,
) -> RuntimeStatus:
    """Compose deterministic readiness without leaking component exceptions."""
    component_tuple = tuple(components)
    names = [component.name for component in component_tuple]
    if len(names) != len(set(names)):
        raise ValueError("readiness component names must be unique")
    ready = lifecycle is LifecycleState.READY and all(
        component.ready for component in component_tuple
    )
    return RuntimeStatus(
        lifecycle=lifecycle,
        readiness=ReadinessState.READY if ready else ReadinessState.NOT_READY,
        checked_at=checked_at,
        components=component_tuple,
    )
