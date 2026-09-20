"""
Geometric Mean.

Ontology: "What is the actual compound return?"
"""
import numpy as np


def geometric_mean(returns: np.ndarray) -> float:
    """
    Compute geometric mean (compound return).

    Parameters
    ----------
    returns : np.ndarray
        Vector of returns (decimal, e.g., 0.10 for 10%).

    Returns
    -------
    float
        Geometric mean return.

    Raises
    ------
    ValueError
        If any return is less than or equal to -100%.

    Examples
    --------
    >>> geometric_mean(np.array([0.10, 0.20, -0.05]))
    0.0784...
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        raise ValueError("Returns must not be empty.")
    if np.any(1 + returns <= 0):
        raise ValueError("Geometric mean undefined: return <= -100% detected.")

    product = float(np.prod(1 + returns))
    n = returns.size
    return product ** (1 / n) - 1
