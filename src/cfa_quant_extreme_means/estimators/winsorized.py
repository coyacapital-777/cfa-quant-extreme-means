"""
Winsorized Mean.

Ontology: "What is the normal value with limited tail risk?"
"""
import numpy as np


def winsorized_mean(
    data: np.ndarray, fraction: float = 0.1
) -> tuple[float, np.ndarray, float, float]:
    """
    Compute winsorized mean.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.
    fraction : float, default=0.1
        Fraction to winsorize from each tail (0, 0.5).

    Returns
    -------
    result : float
        Winsorized mean.
    winsorized_data : np.ndarray
        Data after winsorizing.
    lower_threshold : float
        Lower threshold used.
    upper_threshold : float
        Upper threshold used.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result, _, lo, hi = winsorized_mean(data, 0.10)
    >>> lo, hi
    (3.0, 15.0)
    >>> round(result, 4)
    7.5
    """
    if not 0 <= fraction < 0.5:
        raise ValueError("Fraction must be in [0, 0.5).")

    data = np.asarray(data, dtype=float)
    sorted_data = np.sort(data)
    n = data.size
    k = int(np.floor(fraction * n))

    if k == 0:
        return (
            float(np.mean(sorted_data)),
            sorted_data.copy(),
            float(sorted_data[0]),
            float(sorted_data[-1]),
        )

    lower_threshold = float(sorted_data[k])
    upper_threshold = float(sorted_data[n - k - 1])

    winsorized_data = sorted_data.copy()
    winsorized_data[:k] = lower_threshold
    winsorized_data[n - k :] = upper_threshold

    result = float(np.mean(winsorized_data))
    return result, winsorized_data, lower_threshold, upper_threshold
