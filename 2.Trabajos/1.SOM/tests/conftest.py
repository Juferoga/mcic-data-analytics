"""Pytest fixtures for ETL tests."""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 90.0, 78.5],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_output(tmp_path):
    """Provide a temporary output path."""
    return tmp_path / "output.csv"


@pytest.fixture
def empty_pipeline():
    """Provide an empty Pipeline instance."""
    from analitica.etl import Pipeline

    return Pipeline()
