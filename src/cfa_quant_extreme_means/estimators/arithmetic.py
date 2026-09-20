"""
Arithmetic Mean.

Ontology: "Is the data honest?" — diagnostic for distortion detection.
"""
import numpy as np


def arithmetic_mean(data: np.ndarray) -> tuple[float, str]:
    """
    Compute arithmetic mean with distortion diagnostic.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.

    Returns
    -------
    result : float
        Arithmetic mean.
    interpretation : str
        Distortion signal based on comparison with the median.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result, interp = arithmetic_mean(data)
    >>> round(result, 4)
    17.9
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("Data must not be empty.")

    result = float(np.mean(data))
    med = float(np.median(data))
    gap = abs(result - med)

    if gap > 0.5 * abs(med):
        interpretation = (
            f"RED FLAG: Mean ({result:.4f}) far from median ({med:.4f}). "
            "Possible outlier distortion."
        )
    else:
        interpretation = f"OK: Mean ({result:.4f}) close to median ({med:.4f})."

    return result, interpretation
