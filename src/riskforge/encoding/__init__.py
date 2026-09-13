"""Record-to-tensor encoding for serving.

The application layer consumes the transport-agnostic ``IncidentInputEncoder``
protocol (``riskforge.application.inference_adapter``); this package provides
the concrete HuggingFace tokenizer implementation used with a deployed ONNX
artifact.  ``transformers`` is imported lazily so environments without it
(instead of the heavier modeling extras) stay importable.
"""

from riskforge.encoding.tokenizer_encoder import HFIncidentEncoder

__all__ = ["HFIncidentEncoder"]
