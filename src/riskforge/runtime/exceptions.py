"""Runtime lifecycle failures outside immutable core contracts."""


class RuntimeLifecycleError(RuntimeError):
    """Base class for runtime lifecycle failures."""


class RuntimeStateError(RuntimeLifecycleError):
    """Raised when a lifecycle operation is invalid for the current state."""


class RuntimeStartupError(RuntimeLifecycleError):
    """Raised after failed startup and best-effort rollback."""


class RuntimeShutdownError(RuntimeLifecycleError):
    """Raised after best-effort shutdown encounters failures."""
