"""Tests for data sources."""

import pytest
import pandas as pd
from pathlib import Path

from analitica.etl import CSVSource, ExcelSource, JSONSource, Source
from analitica.etl.exceptions import SourceError


class TestCSVSource:
    """Test cases for CSVSource class."""

    def test_extract_valid_csv(self, tmp_path):
        """CSVSource.extract() reads a valid CSV file."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df.to_csv(csv_path, index=False)

        source = CSVSource(csv_path)
        result = source.extract()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]

    def test_extract_nonexistent_file(self):
        """CSVSource.extract() raises SourceError for missing file."""
        source = CSVSource("nonexistent.csv")

        with pytest.raises(SourceError, match="not found"):
            source.extract()

    def test_extract_with_dtype_kwargs(self, tmp_path):
        """CSVSource passes kwargs to pandas.read_csv."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"a": ["1", "2"]}).to_csv(csv_path, index=False)

        source = CSVSource(csv_path, dtype={"a": int})
        result = source.extract()

        assert result["a"].dtype == int


class TestExcelSource:
    """Test ExcelSource."""

    def test_extract_valid_excel(self, tmp_path):
        """ExcelSource reads a valid Excel file."""
        path = tmp_path / "test.xlsx"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.to_excel(path, index=False)

        source = ExcelSource(path)
        result = source.extract()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_extract_specific_sheet(self, tmp_path):
        """ExcelSource reads a specific sheet."""
        path = tmp_path / "test.xlsx"
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        with pd.ExcelWriter(path) as writer:
            df1.to_excel(writer, sheet_name="First", index=False)
            df2.to_excel(writer, sheet_name="Second", index=False)

        source = ExcelSource(path, sheet_name="Second")
        result = source.extract()

        assert "b" in result.columns

    def test_extract_missing_file(self):
        """ExcelSource raises error for missing file."""
        source = ExcelSource("nonexistent.xlsx")

        with pytest.raises(SourceError, match="not found"):
            source.extract()


class TestJSONSource:
    """Test JSONSource."""

    def test_extract_valid_json(self, tmp_path):
        """JSONSource reads valid JSON."""
        path = tmp_path / "test.json"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.to_json(path, orient="records", indent=2)

        source = JSONSource(path)
        result = source.extract()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_extract_missing_file(self):
        """JSONSource raises error for missing file."""
        source = JSONSource("nonexistent.json")

        with pytest.raises(SourceError, match="not found"):
            source.extract()
