class RiskForgeError(Exception):
    pass

class SchemaValidationError(RiskForgeError):
    pass

class NormalizationError(RiskForgeError):
    pass

class WeakSupervisionConvergenceError(RiskForgeError):
    pass

class ModelInferenceError(RiskForgeError):
    pass

class ONNXExportError(RiskForgeError):
    pass

class QuantizationError(RiskForgeError):
    pass
