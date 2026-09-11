"""Operational analytics: trends, emerging risk, patterns, recommendations.

Pure, deterministic computations over stored incident results.  This
subsystem depends only on core contracts; composers bind it to persistence.
"""

from riskforge.analytics.recommendations import DISCLAIMER, build_recommendations
from riskforge.analytics.service import AnalyticsSummary, compute_summary

__all__ = [
    "DISCLAIMER",
    "AnalyticsSummary",
    "build_recommendations",
    "compute_summary",
]
