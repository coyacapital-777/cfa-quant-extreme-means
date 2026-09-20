"""
Harmonic Mean.

Ontology: "What is the actual purchase price per share?"
"""
import numpy as np


def harmonic_mean(prices: np.ndarray) -> float:
    """
    Compute harmonic mean.

    Parameters
    ----------
    prices : np.ndarray
        Vector of prices (must be positive).

    Returns
    -------
    float
        Harmonic mean.

    Raises
    ------
    ValueError
        If any price is <= 0.

    Examples
    --------
    >>> harmonic_mean(np.array([10, 12, 15]))
    12.0
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        raise ValueError("Prices must not be empty.")
    if np.any(prices <= 0):
        raise ValueError("Harmonic mean requires all prices > 0.")
    return prices.size / float(np.sum(1.0 / prices))


def harmonic_mean_weighted(prices: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted harmonic mean.

    Used when the invested amount differs per period.
    """
    prices = np.asarray(prices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if prices.size != weights.size:
        raise ValueError("Prices and weights must have the same length.")
    if np.any(prices <= 0):
        raise ValueError("Harmonic mean requires all prices > 0.")
    return float(np.sum(weights) / np.sum(weights / prices))
