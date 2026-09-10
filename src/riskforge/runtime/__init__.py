"""Runtime configuration, lifecycle, and dependency-composition contracts."""

from riskforge.runtime.api_adapter import RuntimeReadinessProvider
from riskforge.runtime.asgi import create_managed_app
from riskforge.runtime.contracts import (
    ComponentReadiness,
    DependencyComposer,
    LifecycleComponent,
    LifecycleState,
    ReadinessState,
    RuntimeAssembly,
    RuntimeEnvironment,
    RuntimeSettings,
    RuntimeStatus,
    compose_readiness,
)
from riskforge.runtime.exceptions import (
    RuntimeLifecycleError,
    RuntimeShutdownError,
    RuntimeStartupError,
    RuntimeStateError,
)
from riskforge.runtime.lifecycle import RuntimeManager
from riskforge.runtime.persistence_adapter import PostgresPoolLifecycle
from riskforge.runtime.workflow_adapters import PersistenceWorkflowReader, ReviewServiceAdapter

__all__ = [
    "ComponentReadiness",
    "DependencyComposer",
    "LifecycleComponent",
    "LifecycleState",
    "PersistenceWorkflowReader",
    "PostgresPoolLifecycle",
    "ReadinessState",
    "ReviewServiceAdapter",
    "RuntimeAssembly",
    "RuntimeEnvironment",
    "RuntimeLifecycleError",
    "RuntimeManager",
    "RuntimeReadinessProvider",
    "RuntimeSettings",
    "RuntimeShutdownError",
    "RuntimeStartupError",
    "RuntimeStateError",
    "RuntimeStatus",
    "compose_readiness",
    "create_managed_app",
]
