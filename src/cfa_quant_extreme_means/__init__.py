"""
CFA Level 1 Quant — Extreme Value Means.

A toolkit for computing robust estimators from stock returns
with the Griffin-Singer framework: classify outliers first
(Error / Noise / Signal), then choose the appropriate estimator.
"""

__version__ = "0.1.0"

from .classification.outlier_classifier import (
    AnomalyClassification,
    AnomalyType,
    classify_outliers,
    recommend_estimator,
)
from .comparison.mean_comparator import MeanComparison, compare_all_means
from .estimators.arithmetic import arithmetic_mean
from .estimators.geometric import geometric_mean
from .estimators.harmonic import harmonic_mean
from .estimators.iqm import iqm_mean
from .estimators.median import median_mean
from .estimators.mom import mom_mean
from .estimators.trimmed import trimmed_mean
from .estimators.winsorized import winsorized_mean

__all__ = [
    "arithmetic_mean",
    "geometric_mean",
    "harmonic_mean",
    "trimmed_mean",
    "winsorized_mean",
    "mom_mean",
    "iqm_mean",
    "median_mean",
    "classify_outliers",
    "recommend_estimator",
    "compare_all_means",
    "AnomalyType",
    "AnomalyClassification",
    "MeanComparison",
]
