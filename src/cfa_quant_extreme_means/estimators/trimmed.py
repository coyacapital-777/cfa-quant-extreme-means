"""
Trimmed Mean.

Ontology: "What is the normal value without noise?"
"""
import numpy as np


def trimmed_mean(
    data: np.ndarray, fraction: float = 0.1
) -> tuple[float, np.ndarray, int]:
    """
    Compute trimmed mean.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.
    fraction : float, default=0.1
        Fraction to trim from each tail (0, 0.5).

    Returns
    -------
    result : float
        Trimmed mean.
    trimmed_data : np.ndarray
        Data after trimming.
    n_trimmed : int
        Number of observations trimmed from each tail.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result, _, k = trimmed_mean(data, 0.10)
    >>> k
    1
    >>> round(result, 4)
    7.125
    """
    if not 0 <= fraction < 0.5:
        raise ValueError("Fraction must be in [0, 0.5).")

    data = np.asarray(data, dtype=float)
    sorted_data = np.sort(data)
    n = data.size
    k = int(np.floor(fraction * n))

    if 2 * k >= n:
        raise ValueError("Too many data points would be trimmed.")

    trimmed_data = sorted_data[k : n - k]
    result = float(np.mean(trimmed_data))
    return result, trimmed_data, k
