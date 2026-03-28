"""Tests for transformers/encoders module."""

import pytest
import numpy as np
import pandas as pd

from analitica.transformers import (
    BaseEncoder,
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder,
    HashEncoder,
)


class TestLabelEncoder:
    """Test LabelEncoder."""

    def test_fit_transform_basic(self):
        """LabelEncoder maps categories to integers."""
        data = pd.DataFrame({"color": ["red", "blue", "green", "red"]})

        encoder = LabelEncoder()
        result = encoder.fit_transform(data)

        # Should have integer labels
        assert result["color"].dtype in [np.int32, np.int64, int]
        assert len(result) == 4

    def test_fit_transform_specific_columns(self):
        """LabelEncoder works with specific columns."""
        data = pd.DataFrame(
            {"color": ["red", "blue", "green"], "size": ["S", "M", "L"]}
        )

        encoder = LabelEncoder(columns=["color"])
        result = encoder.fit_transform(data)

        assert "color" in result.columns
        assert "size" in result.columns  # Unchanged

    def test_unseen_category_warning(self):
        """LabelEncoder warns on unseen categories."""
        train_data = pd.DataFrame({"cat": ["A", "B", "C"]})
        test_data = pd.DataFrame({"cat": ["A", "D", "E"]})  # D, E unseen

        encoder = LabelEncoder(handle_unknown="warn")
        encoder.fit(train_data)

        with pytest.warns(UserWarning):
            result = encoder.transform(test_data)

        # D and E should be -1
        assert result["cat"].iloc[1] == -1
        assert result["cat"].iloc[2] == -1


class TestOneHotEncoder:
    """Test OneHotEncoder."""

    def test_fit_transform_basic(self):
        """OneHotEncoder creates binary columns."""
        data = pd.DataFrame({"city": ["NY", "LA", "SF", "NY"]})

        encoder = OneHotEncoder()
        result = encoder.fit_transform(data)

        # Should have one-hot columns, original dropped
        assert "city" not in result.columns
        assert "city_NY" in result.columns
        assert "city_LA" in result.columns
        assert "city_SF" in result.columns

        # Values should be 0 or 1
        assert set(result["city_NY"].unique()).issubset({0, 1})

    def test_drop_first(self):
        """OneHotEncoder with drop_first removes one category."""
        data = pd.DataFrame({"color": ["red", "blue", "green"]})

        encoder = OneHotEncoder(drop_first=True)
        result = encoder.fit_transform(data)

        # Should have 2 columns instead of 3
        color_cols = [c for c in result.columns if c.startswith("color_")]
        assert len(color_cols) == 2


class TestTargetEncoder:
    """Test TargetEncoder."""

    def test_fit_transform_basic(self):
        """TargetEncoder encodes with target mean."""
        data = pd.DataFrame(
            {"city": ["NY", "NY", "LA", "SF"], "amount": [100, 200, 150, 300]}
        )
        y = pd.Series([1, 1, 0, 1])  # NY=1.0, LA=0.0, SF=1.0

        encoder = TargetEncoder()
        result = encoder.fit_transform(data, y)

        # Should have encoded city column
        assert "city" in result.columns
        assert result["city"].dtype == float
        # NY should be higher than LA
        assert result["city"].iloc[0] > result["city"].iloc[2]

    def test_requires_target(self):
        """TargetEncoder requires y parameter."""
        data = pd.DataFrame({"cat": ["A", "B", "C"]})

        encoder = TargetEncoder()

        with pytest.raises(ValueError, match="requires target"):
            encoder.fit(data)


class TestHashEncoder:
    """Test HashEncoder."""

    def test_fit_transform_basic(self):
        """HashEncoder creates hash columns."""
        data = pd.DataFrame(
            {"url": ["http://a.com", "http://b.com", "http://c.com"], "id": [1, 2, 3]}
        )

        encoder = HashEncoder(columns=["url"], n_bins=4, n_functions=2)
        result = encoder.fit_transform(data)

        # Should have hash columns
        assert "url" not in result.columns
        assert "url_hash_0" in result.columns
        assert "url_hash_1" in result.columns

        # Values should be in range [0, n_bins)
        assert result["url_hash_0"].max() < 4

    def test_deterministic(self):
        """HashEncoder is deterministic."""
        data = pd.DataFrame({"url": ["http://test.com"]})

        encoder = HashEncoder(n_bins=8, n_functions=2)
        result1 = encoder.fit_transform(data)
        result2 = encoder.fit_transform(data)

        # Same input -> same output
        assert result1["url_hash_0"].iloc[0] == result2["url_hash_0"].iloc[0]


class TestIntegration:
    """Integration tests with ETL pipeline."""

    def test_pipeline_with_encoders(self, tmp_path):
        """Encoders work in ETL pipeline."""
        from analitica.etl import Pipeline, CSVSource, CSVDestination

        # Create test data
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        data = pd.DataFrame(
            {"id": [1, 2, 3], "category": ["A", "B", "A"], "value": [10, 20, 30]}
        )
        data.to_csv(input_path, index=False)

        # Run pipeline with encoder
        pipeline = (
            Pipeline()
            .extract_from(CSVSource(input_path))
            .add_transformer(LabelEncoder(columns=["category"]))
            .load_to(CSVDestination(output_path))
        )
        result = pipeline.run()

        assert result["category"].dtype in [np.int32, np.int64, int]
        assert len(result) == 3
