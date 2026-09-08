"""RiskForge Metrics & Asset Risk Analytics subsystem.

Aggregates ModelInferenceResult records into per-asset AssetRiskSummary objects,
computing SIF Precursor Density (SPD) scores and tracking recurrent failed barriers.
"""

from riskforge.metrics.aggregator import MetricsAggregator, compute_asset_risk_summary

__all__ = ["MetricsAggregator", "compute_asset_risk_summary"]
