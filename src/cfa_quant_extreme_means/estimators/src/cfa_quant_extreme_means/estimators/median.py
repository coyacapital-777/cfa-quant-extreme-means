"""
Median.

Ontology: "What is the center that cannot be disturbed by outliers?"
"""
import numpy as np


def median_mean(data: np.ndarray) -> tuple[float, str]:
    """
    Compute median as a robust benchmark.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.

    Returns
    -------
    result : float
        Median.
    interpretation : str
        Human-readable description.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result, note = median_mean(data)
    >>> result
    6.5
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("Data must not be empty.")

    result = float(np.median(data))
    return result, f"Median: {result:.4f} (robust to outliers)"
