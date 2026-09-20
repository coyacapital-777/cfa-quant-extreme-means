"""Unit tests for outlier classification and comparison."""
import numpy as np

from cfa_quant_extreme_means.classification.outlier_classifier import (
    AnomalyType,
    classify_outliers,
    recommend_estimator,
)
from cfa_quant_extreme_means.comparison.mean_comparator import compare_all_means


# ---------- Classification ----------


def test_classify_returns_all_indices():
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)
    results = classify_outliers(data)
    assert len(results) == data.size
    assert [r.index for r in results] == list(range(data.size))


def test_120_classified_as_error():
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)
    results = classify_outliers(data)
    assert results[-1].anomaly_type == AnomalyType.ERROR


def test_normal_data_all_normal():
    data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
    results = classify_outliers(data)
    types = {r.anomaly_type for r in results}
    assert AnomalyType.ERROR not in types


def test_classify_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        classify_outliers(np.array([]))


# ---------- Recommendation ----------


def test_recommend_estimator_with_error():
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)
    assert recommend_estimator(data) == "trimmed_mean"


def test_recommend_estimator_clean_data():
    data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
    assert recommend_estimator(data) == "arithmetic_mean"


# ---------- Comparison ----------


def test_compare_all_means_returns_dataclass():
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)
    result = compare_all_means(data)
    assert result.arithmetic > result.trimmed_10
    assert result.trimmed_10 == 7.125
    assert result.winsorized_10 == 7.5
    assert result.median == 6.5
    assert result.recommendation == "trimmed_mean"


def test_compare_all_means_gap():
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)
    result = compare_all_means(data)
    assert result.gap_arith_geo > 0.10  # high distortion


def test_compare_all_means_skewness():
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)
    result = compare_all_means(data)
    assert result.skewness > 1.0  # highly skewed


def test_compare_all_means_requires_four_observations():
    import pytest

    with pytest.raises(ValueError):
        compare_all_means(np.array([1, 2, 3]))
