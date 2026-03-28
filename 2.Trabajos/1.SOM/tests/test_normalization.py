"""Tests for normalization module."""

import pytest
import numpy as np
import pandas as pd

from analitica.normalization import (
    BaseNormalizer,
    MinMaxScaler,
    ZScoreScaler,
    RobustScaler,
    LogTransformer,
    PowerTransformer,
)


class TestMinMaxScaler:
    """Test MinMaxScaler."""

    def test_fit_transform_basic(self):
        """MinMaxScaler scales to [0, 1]."""
        data = pd.DataFrame({"a": [0, 50, 100], "b": [10, 20, 30]})

        scaler = MinMaxScaler()
        result = scaler.fit_transform(data)

        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0
        assert result["b"].min() == 0.0
        assert result["b"].max() == 1.0

    def test_fit_transform_specific_columns(self):
        """MinMaxScaler works with specific columns."""
        data = pd.DataFrame(
            {"a": [0, 50, 100], "b": [10, 20, 30], "c": ["x", "y", "z"]}
        )

        scaler = MinMaxScaler(columns=["a"])
        result = scaler.fit_transform(data)

        assert result["a"].max() == 1.0
        assert result["b"].tolist() == [10, 20, 30]  # Unchanged

    def test_constant_column(self):
        """MinMaxScaler handles constant columns."""
        data = pd.DataFrame({"a": [5, 5, 5], "b": [0, 50, 100]})

        scaler = MinMaxScaler()
        result = scaler.fit_transform(data)

        assert (result["a"] == 0).all()  # Constant → 0
        assert result["b"].max() == 1.0


class TestZScoreScaler:
    """Test ZScoreScaler."""

    def test_fit_transform_basic(self):
        """ZScoreScaler standardizes to mean=0, std=1."""
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        scaler = ZScoreScaler()
        result = scaler.fit_transform(data)

        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["a"].std() - 1.0) < 1e-10

    def test_constant_column(self):
        """ZScoreScaler handles constant columns."""
        data = pd.DataFrame({"a": [5, 5, 5]})

        scaler = ZScoreScaler()
        result = scaler.fit_transform(data)

        assert (result["a"] == 0).all()


class TestRobustScaler:
    """Test RobustScaler."""

    def test_fit_transform_basic(self):
        """RobustScaler uses median and IQR."""
        data = pd.DataFrame({"a": [1, 2, 3, 4, 100]})  # 100 is outlier

        scaler = RobustScaler()
        result = scaler.fit_transform(data)

        # Median should be at 0
        median_idx = len(result) // 2
        assert abs(result["a"].iloc[median_idx]) < 1.0  # Close to 0

    def test_zero_iqr(self):
        """RobustScaler handles zero IQR."""
        data = pd.DataFrame({"a": [5, 5, 5], "b": [1, 2, 3]})

        scaler = RobustScaler()
        result = scaler.fit_transform(data)

        assert (result["a"] == 0).all()


class TestLogTransformer:
    """Test LogTransformer."""

    def test_fit_transform_positive(self):
        """LogTransformer works with positive values."""
        data = pd.DataFrame({"a": [1, 10, 100, 1000]})

        transformer = LogTransformer()
        result = transformer.fit_transform(data)

        # Should be monotonically increasing
        assert result["a"].iloc[0] < result["a"].iloc[1]
        assert result["a"].iloc[2] < result["a"].iloc[3]

    def test_fit_transform_negative(self):
        """LogTransformer handles negative values."""
        data = pd.DataFrame({"a": [-5, 0, 5, 10]})

        transformer = LogTransformer()
        result = transformer.fit_transform(data)

        # All values should be finite
        assert np.isfinite(result["a"]).all()

    def test_fit_transform_zeros(self):
        """LogTransformer handles zeros."""
        data = pd.DataFrame({"a": [0, 1, 2, 3]})

        transformer = LogTransformer()
        result = transformer.fit_transform(data)

        assert np.isfinite(result["a"]).all()


class TestPowerTransformer:
    """Test PowerTransformer."""

    def test_fit_transform_basic(self):
        """PowerTransformer normalizes skewed data."""
        data = pd.DataFrame({"a": [1, 2, 4, 8, 16]})  # Log-normal distribution

        transformer = PowerTransformer()
        result = transformer.fit_transform(data)

        # Result should be approximately normal
        assert abs(result["a"].mean()) < 1.0
        assert abs(result["a"].std() - 1.0) < 1.0


class TestIntegration:
    """Integration tests with ETL pipeline."""

    def test_pipeline_with_normalizer(self, tmp_path):
        """Normalizers work in ETL pipeline."""
        from analitica.etl import Pipeline, CSVSource, CSVDestination

        # Create test data
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        data = pd.DataFrame(
            {"id": [1, 2, 3], "value": [10, 50, 100], "name": ["a", "b", "c"]}
        )
        data.to_csv(input_path, index=False)

        # Run pipeline with normalizer
        pipeline = (
            Pipeline()
            .extract_from(CSVSource(input_path))
            .add_transformer(MinMaxScaler(columns=["value"]))
            .load_to(CSVDestination(output_path))
        )
        result = pipeline.run()

        assert result["value"].max() == 1.0
        assert "name" in result.columns
