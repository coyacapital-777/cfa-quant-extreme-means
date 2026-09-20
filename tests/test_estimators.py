"""Unit tests for all estimators."""
import numpy as np
import pytest

from cfa_quant_extreme_means.estimators.arithmetic import arithmetic_mean
from cfa_quant_extreme_means.estimators.geometric import geometric_mean
from cfa_quant_extreme_means.estimators.harmonic import (
    harmonic_mean,
    harmonic_mean_weighted,
)
from cfa_quant_extreme_means.estimators.iqm import iqm_mean
from cfa_quant_extreme_means.estimators.median import median_mean
from cfa_quant_extreme_means.estimators.mom import mom_mean
from cfa_quant_extreme_means.estimators.trimmed import trimmed_mean
from cfa_quant_extreme_means.estimators.winsorized import winsorized_mean


@pytest.fixture
def cfa_data() -> np.ndarray:
    """CFA Level 1 sample data with one extreme outlier."""
    return np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)


# ---------- Arithmetic ----------


def test_arithmetic_mean_value(cfa_data):
    result, _ = arithmetic_mean(cfa_data)
    assert abs(result - 17.9) < 1e-6


def test_arithmetic_mean_red_flag(cfa_data):
    _, note = arithmetic_mean(cfa_data)
    assert "RED FLAG" in note


def test_arithmetic_mean_empty_raises():
    with pytest.raises(ValueError):
        arithmetic_mean(np.array([]))


# ---------- Geometric ----------


def test_geometric_mean_value():
    returns = np.array([0.10, 0.20, -0.05])
    expected = (1.10 * 1.20 * 0.95) ** (1 / 3) - 1
    assert abs(geometric_mean(returns) - expected) < 1e-6


def test_geometric_mean_invalid_raises():
    with pytest.raises(ValueError):
        geometric_mean(np.array([-1.5, 0.10]))


def test_geometric_mean_empty_raises():
    with pytest.raises(ValueError):
        geometric_mean(np.array([]))


# ---------- Harmonic ----------


def test_harmonic_mean_value():
    prices = np.array([10, 12, 15], dtype=float)
    expected = 3 / (1 / 10 + 1 / 12 + 1 / 15)
    assert abs(harmonic_mean(prices) - expected) < 1e-6


def test_harmonic_mean_negative_raises():
    with pytest.raises(ValueError):
        harmonic_mean(np.array([-1, 2, 3]))


def test_harmonic_mean_zero_raises():
    with pytest.raises(ValueError):
        harmonic_mean(np.array([0, 2, 3]))


def test_harmonic_mean_weighted():
    prices = np.array([10, 12, 15], dtype=float)
    weights = np.array([1, 1, 1], dtype=float)
    assert abs(harmonic_mean_weighted(prices, weights) - harmonic_mean(prices)) < 1e-9


# ---------- Trimmed ----------


def test_trimmed_mean_10(cfa_data):
    result, trimmed, k = trimmed_mean(cfa_data, 0.10)
    assert k == 1
    assert 2 not in trimmed
    assert 120 not in trimmed
    expected = np.mean([3, 4, 5, 6, 7, 8, 9, 15])
    assert abs(result - expected) < 1e-6


def test_trimmed_mean_20(cfa_data):
    result, trimmed, k = trimmed_mean(cfa_data, 0.20)
    assert k == 2
    expected = np.mean([4, 5, 6, 7, 8, 9])
    assert abs(result - expected) < 1e-6


def test_trimmed_mean_invalid_fraction():
    with pytest.raises(ValueError):
        trimmed_mean(np.array([1, 2, 3, 4]), 0.6)


def test_trimmed_mean_too_many_trimmed():
    with pytest.raises(ValueError):
        trimmed_mean(np.array([1, 2, 3, 4]), 0.49)


# ---------- Winsorized ----------


def test_winsorized_mean_10(cfa_data):
    result, wins, lo, hi = winsorized_mean(cfa_data, 0.10)
    assert lo == 3.0
    assert hi == 15.0
    expected = np.mean([3, 3, 4, 5, 6, 7, 8, 9, 15, 15])
    assert abs(result - expected) < 1e-6


def test_winsorized_preserves_n(cfa_data):
    _, wins, _, _ = winsorized_mean(cfa_data, 0.20)
    assert wins.size == cfa_data.size


def test_winsorized_mean_invalid_fraction():
    with pytest.raises(ValueError):
        winsorized_mean(np.array([1, 2, 3, 4]), -0.1)


# ---------- MOM ----------


def test_mom_detects_outlier(cfa_data):
    result, mask, bounds, n_out = mom_mean(cfa_data)
    assert mask[-1] == True  # 120 is outlier
    assert result < 20
    assert n_out >= 1


def test_mom_requires_three_observations():
    with pytest.raises(ValueError):
        mom_mean(np.array([1, 2]))


# ---------- IQM ----------


def test_iqm_value(cfa_data):
    result, middle = iqm_mean(cfa_data)
    assert 4 < result < 15
    assert 2 not in middle
    assert 120 not in middle


def test_iqm_requires_four_observations():
    with pytest.raises(ValueError):
        iqm_mean(np.array([1, 2, 3]))


# ---------- Median ----------


def test_median_value(cfa_data):
    result, note = median_mean(cfa_data)
    assert result == 6.5
    assert "robust" in note.lower()


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median_mean(np.array([]))
