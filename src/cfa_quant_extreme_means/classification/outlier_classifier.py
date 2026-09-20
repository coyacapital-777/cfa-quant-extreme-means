"""
Outlier Classification: Error, Noise, or Signal.

Framework: Griffin-Singer
- Error: z-score > error_threshold -> trim
- Noise: z-score > noise_threshold -> winsorize
- Signal: outlier within thresholds -> retain
- Normal: within IQR bounds
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class AnomalyType(Enum):
    ERROR = "Error"
    NOISE = "Noise"
    SIGNAL = "Signal"
    NORMAL = "Normal"


@dataclass
class AnomalyClassification:
    index: int
    value: float
    anomaly_type: AnomalyType
    z_score: float
    reason: str


def classify_outliers(
    data: np.ndarray,
    error_threshold: float = 5.0,
    noise_threshold: float = 3.0,
) -> list[AnomalyClassification]:
    """
    Classify each observation as Error, Noise, Signal, or Normal.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.
    error_threshold : float, default=5.0
        Z-score threshold for classifying as Error.
    noise_threshold : float, default=3.0
        Z-score threshold for classifying as Noise.

    Returns
    -------
    list[AnomalyClassification]
        Classification per observation.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> results = classify_outliers(data)
    >>> results[-1].anomaly_type
    <AnomalyType.ERROR: 'Error'>
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("Data must not be empty.")

    mu = float(np.mean(data))
    sigma = float(np.std(data))

    results: list[AnomalyClassification] = []
    for i, value in enumerate(data):
        z_score = abs(value - mu) / sigma if sigma > 0 else 0.0

        if z_score > error_threshold:
            anomaly_type = AnomalyType.ERROR
            reason = f"Z-score {z_score:.2f} > {error_threshold} (likely error)"
        elif z_score > noise_threshold:
            anomaly_type = AnomalyType.NOISE
            reason = f"Z-score {z_score:.2f} > {noise_threshold} (noise)"
        elif z_score > 1.5:
            anomaly_type = AnomalyType.SIGNAL
            reason = f"Outlier but within thresholds (z={z_score:.2f})"
        else:
            anomaly_type = AnomalyType.NORMAL
            reason = "Within normal bounds"

        results.append(
            AnomalyClassification(
                index=i,
                value=float(value),
                anomaly_type=anomaly_type,
                z_score=z_score,
                reason=reason,
            )
        )

    return results


def recommend_estimator(data: np.ndarray) -> str:
    """
    Recommend the appropriate estimator based on outlier classification.

    Returns
    -------
    str
        Name of recommended estimator.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> recommend_estimator(data)
    'trimmed_mean'
    """
    classifications = classify_outliers(data)
    types = [c.anomaly_type for c in classifications]

    n_error = types.count(AnomalyType.ERROR)
    n_noise = types.count(AnomalyType.NOISE)
    n_signal = types.count(AnomalyType.SIGNAL)

    if n_error > 0:
        return "trimmed_mean"
    if n_noise > 0:
        return "winsorized_mean"
    if n_signal > 0:
        return "arithmetic_mean + separate analysis"
    return "arithmetic_mean"
4. Commit message:

text
feat: add outlier classifier (Error/Noise/Signal)
5. Klik "Commit new file"

File 13: comparison/__init__.py
1. Klik "Add file" -> "Create new file"

2. Nama file:

text
src/cfa_quant_extreme_means/comparison/__init__.py
3. Paste isi:

python
"""Comparison of all extreme value means."""
4. Commit message:

text
feat: add comparison subpackage init
5. Klik "Commit new file"

File 14: comparison/mean_comparator.py
1. Klik "Add file" -> "Create new file"

2. Nama file:

text
src/cfa_quant_extreme_means/comparison/mean_comparator.py
3. Paste isi:

python
"""
Compare all extreme value means.

Framework: Griffin-Singer
- Arithmetic vs Geometric gap = distortion signal
- Trimmed vs Winsorized = error vs noise
- MOM vs IQM = adaptive vs consensus
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..classification.outlier_classifier import recommend_estimator
from ..estimators.arithmetic import arithmetic_mean
from ..estimators.geometric import geometric_mean
from ..estimators.harmonic import harmonic_mean
from ..estimators.iqm import iqm_mean
from ..estimators.median import median_mean
from ..estimators.mom import mom_mean
from ..estimators.trimmed import trimmed_mean
from ..estimators.winsorized import winsorized_mean


@dataclass
class MeanComparison:
    arithmetic: float
    arithmetic_note: str
    geometric: float
    harmonic: float
    trimmed_10: float
    trimmed_20: float
    winsorized_10: float
    winsorized_20: float
    mom: float
    iqm: float
    median: float
    gap_arith_geo: float
    skewness: float
    recommendation: str
    warnings: list[str] = field(default_factory=list)


def compare_all_means(data: np.ndarray) -> MeanComparison:
    """
    Compare all estimators and generate a recommendation.

    Parameters
    ----------
    data : np.ndarray
        Vector of stock returns.

    Returns
    -------
    MeanComparison
        Structured comparison result.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result = compare_all_means(data)
    >>> round(result.arithmetic, 4)
    17.9
    >>> round(result.trimmed_10, 4)
    7.125
    >>> round(result.winsorized_10, 4)
    7.5
    >>> result.recommendation
    'trimmed_mean'
    """
    data = np.asarray(data, dtype=float)
    if data.size < 4:
        raise ValueError("At least 4 observations are required.")

    warnings: list[str] = []

    arith, arith_note = arithmetic_mean(data)

    try:
        geo = geometric_mean(data) if np.all(data > -1) else float("nan")
    except ValueError:
        geo = float("nan")
        warnings.append("Geometric mean undefined (return <= -100%).")

    try:
        harm = harmonic_mean(data) if np.all(data > 0) else float("nan")
    except ValueError:
        harm = float("nan")
        warnings.append("Harmonic mean undefined (non-positive values).")

    trimmed_10, _, _ = trimmed_mean(data, 0.10)
    trimmed_20, _, _ = trimmed_mean(data, 0.20)
    winsorized_10, _, _, _ = winsorized_mean(data, 0.10)
    winsorized_20, _, _, _ = winsorized_mean(data, 0.20)
    mom, _, _, _ = mom_mean(data)
    iqm, _ = iqm_mean(data)
    med, _ = median_mean(data)

    gap = arith - geo if not np.isnan(geo) else float("nan")
    skew = float(stats.skew(data))

    rec = recommend_estimator(data)

    return MeanComparison(
        arithmetic=arith,
        arithmetic_note=arith_note,
        geometric=geo,
        harmonic=harm,
        trimmed_10=trimmed_10,
        trimmed_20=trimmed_20,
        winsorized_10=winsorized_10,
        winsorized_20=winsorized_20,
        mom=mom,
        iqm=iqm,
        median=med,
        gap_arith_geo=gap,
        skewness=skew,
        recommendation=rec,
        warnings=warnings,
    )
