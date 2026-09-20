"""
Interquartile Mean (IQM).

Ontology: "What is the consensus of the middle 50%?"
"""
import numpy as np


def iqm_mean(data: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute Interquartile Mean.

    Trims 25% of lowest and 25% of highest data, then takes the
    mean of the middle 50%.

    Parameters
    ----------
    data : np.ndarray
        Vector of observations.

    Returns
    -------
    result : float
        IQM estimate.
    middle_data : np.ndarray
        Data within the interquartile range.

    Examples
    --------
    >>> data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120])
    >>> result, middle = iqm_mean(data)
    >>> round(result, 4)
    6.5
    """
    data = np.asarray(data, dtype=float)
    if data.size < 4:
        raise ValueError("At least 4 observations are required for IQM.")

    q1, q3 = np.percentile(data, [25, 75])
    middle_data = data[(data >= q1) & (data <= q3)]

    if middle_data.size < 3:
        raise ValueError("Too few data points remain after trimming.")

    return float(np.mean(middle_data)), middle_data
