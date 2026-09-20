"""
Modified One-Step M-estimator (Skipped Estimator / MOM).

Ontology: "Where does reality end and contamination begin?"
"""
import numpy as np


def mom_mean(
    data: np.ndarray, iqr_multiplier: float = 1.5
) -> tuple[float, np.ndarray, tuple[float, float], int]:
    """
    Compute Modified One-Step M-estimator (MOM).

    Empirically determines how many observations to trim based on
    the boxplot (IQR) rule.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.
    iqr_multiplier : float, default=1.5
        IQR multiplier for outlier bounds.

    Returns
    -------
    result : float
        MOM estimate.
    outlier_mask : np.ndarray
        Boolean mask of outliers.
    bounds : tuple
        (lower, upper) bounds.
    n_trimmed : int
        Number of outliers removed.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result, mask, bounds, n_out = mom_mean(data)
    >>> n_out
    1
    >>> round(result, 4)
    6.5
    """
    data = np.asarray(data, dtype=float)
    if data.size < 3:
        raise ValueError("At least 3 observations are required.")

    q1, q3 = np.percentile(data, [25, 75])
    iqr_val = q3 - q1

    lower = q1 - iqr_multiplier * iqr_val
    upper = q3 + iqr_multiplier * iqr_val

    outlier_mask = (data < lower) | (data > upper)
    filtered = data[~outlier_mask]

    if filtered.size < 3:
        raise ValueError("Too many outliers; insufficient data remains.")

    result = float(np.mean(filtered))
    return result, outlier_mask, (float(lower), float(upper)), int(outlier_mask.sum())
