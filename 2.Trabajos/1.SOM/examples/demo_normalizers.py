#!/usr/bin/env python3
"""Demo: Data Normalizers in Analitica."""

import pandas as pd
import numpy as np

from analitica.normalization import (
    MinMaxScaler,
    ZScoreScaler,
    RobustScaler,
    LogTransformer,
    PowerTransformer,
)


def main():
    print("=" * 60)
    print("Analitica - Data Normalizers Demo")
    print("=" * 60)

    # Sample data with different characteristics
    data = pd.DataFrame(
        {
            "age": [18, 25, 35, 45, 65, 22, 30, 50],
            "income": [15000, 35000, 75000, 120000, 500000, 28000, 45000, 90000],
            "score": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.8],
            "constant": [5, 5, 5, 5, 5, 5, 5, 5],
            "has_outlier": [1, 2, 3, 4, 100, 2, 3, 4],
        }
    )

    print("\n📊 Original Data:")
    print(data.describe().round(2))

    # MinMaxScaler
    print("\n" + "-" * 40)
    print("🔄 MinMaxScaler (scale to [0, 1])")
    print("-" * 40)
    minmax = MinMaxScaler()
    result_minmax = minmax.fit_transform(data[["age", "score"]])
    print(result_minmax.round(3))

    # ZScoreScaler
    print("\n" + "-" * 40)
    print("🔄 ZScoreScaler (mean=0, std=1)")
    print("-" * 40)
    zscore = ZScoreScaler()
    result_zscore = zscore.fit_transform(data[["age", "income"]])
    print(result_zscore.round(3))

    # RobustScaler
    print("\n" + "-" * 40)
    print("🔄 RobustScaler (median/IQR - outlier resistant)")
    print("-" * 40)
    robust = RobustScaler()
    result_robust = robust.fit_transform(data[["has_outlier"]])
    print(result_robust.round(3))

    # LogTransformer
    print("\n" + "-" * 40)
    print("🔄 LogTransformer (reduce skewness)")
    print("-" * 40)
    log = LogTransformer()
    result_log = log.fit_transform(data[["income"]])
    print(result_log.round(3))

    # PowerTransformer
    print("\n" + "-" * 40)
    print("🔄 PowerTransformer (Yeo-Johnson)")
    print("-" * 40)
    power = PowerTransformer()
    result_power = power.fit_transform(data[["income", "has_outlier"]])
    print(result_power.round(3))

    print("\n" + "=" * 60)
    print("✅ All normalizers working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
