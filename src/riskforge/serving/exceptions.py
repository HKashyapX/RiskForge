"""Stable exception taxonomy for external model-serving boundaries."""


class ServingError(RuntimeError):
    """Base class for failures exposed by the serving subsystem."""


class ArtifactLoadingError(ServingError):
    """Raised when a model artifact cannot be loaded safely."""


class InputCompatibilityError(ServingError, ValueError):
    """Raised when inference inputs are incompatible with the model contract."""


class InferenceFailureError(ServingError):
    """Raised when the runtime fails while executing a model."""


class InvalidOutputError(ServingError, ValueError):
    """Raised when model outputs violate the serving contract."""


class WarmupError(ServingError):
    """Raised when startup warm-up cannot complete successfully."""


class RequestQueueFullError(ServingError):
    """Raised when bounded request capacity has been exhausted."""


class BatcherClosedError(ServingError):
    """Raised when work is submitted after batcher shutdown begins."""
