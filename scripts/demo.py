"""
Demo: CFA Level 1 Quant — Extreme Value Means.

Run:
    uv run python scripts/demo.py
    # or
    make demo
"""
import numpy as np

from cfa_quant_extreme_means.classification.outlier_classifier import (
    classify_outliers,
)
from cfa_quant_extreme_means.comparison.mean_comparator import compare_all_means


def main() -> None:
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 15, 120], dtype=float)

    print("=" * 64)
    print("CFA LEVEL 1 QUANT — EXTREME VALUE MEANS")
    print("=" * 64)
    print(f"Data: {data.tolist()}")
    print(f"n    = {data.size}")

    print("\n" + "-" * 64)
    print("OUTLIER CLASSIFICATION (Error / Noise / Signal)")
    print("-" * 64)
    for c in classify_outliers(data):
        print(
            f"  [{c.index:>2}] {c.value:>7.2f}  "
            f"{c.anomaly_type.value:>6}  z={c.z_score:>5.2f}  {c.reason}"
        )

    result = compare_all_means(data)

    print("\n" + "-" * 64)
    print("ESTIMATOR COMPARISON")
    print("-" * 64)
    print(f"  Arithmetic Mean    : {result.arithmetic:>10.4f}   Distortion diagnostic")
    print(f"  Geometric Mean     : {result.geometric:>10.4f}   Compound performance")
    print(f"  Harmonic Mean      : {result.harmonic:>10.4f}   Purchase price")
    print(f"  Trimmed Mean 10%   : {result.trimmed_10:>10.4f}   Value without noise")
    print(f"  Trimmed Mean 20%   : {result.trimmed_20:>10.4f}")
    print(f"  Winsorized 10%     : {result.winsorized_10:>10.4f}   Limited tail risk")
    print(f"  Winsorized 20%     : {result.winsorized_20:>10.4f}")
    print(f"  MOM                : {result.mom:>10.4f}   Empirical bounds")
    print(f"  IQM                : {result.iqm:>10.4f}   Middle 50% consensus")
    print(f"  Median             : {result.median:>10.4f}   Robust center")

    print("\n" + "-" * 64)
    print("DIAGNOSTICS")
    print("-" * 64)
    print(f"  Gap Arithmetic-Geometric : {result.gap_arith_geo:>10.4f}")
    if result.gap_arith_geo > 0.10:
        print("    -> HIGH distortion (outliers dominate)")
    elif result.gap_arith_geo > 0.05:
        print("    -> MODERATE distortion")
    else:
        print("    -> LOW distortion")

    print(f"  Skewness                 : {result.skewness:>10.4f}")
    print(f"  Recommendation           : {result.recommendation}")

    if result.warnings:
        print("\n  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
