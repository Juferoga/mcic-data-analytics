"""Tests for data destinations."""

import pytest
import pandas as pd
from pathlib import Path

from analitica.etl import CSVDestination, ExcelDestination, JSONDestination, Destination
from analitica.etl.exceptions import DestinationError


class TestCSVDestination:
    """Test cases for CSVDestination class."""

    def test_save_dataframe(self, tmp_path):
        """CSVDestination.save() writes DataFrame to CSV."""
        output_path = tmp_path / "output.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        dest = CSVDestination(output_path)
        dest.save(df)

        assert output_path.exists()
        result = pd.read_csv(output_path)
        assert len(result) == 2

    def test_save_creates_parent_directories(self, tmp_path):
        """CSVDestination.save() creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dir" / "output.csv"
        df = pd.DataFrame({"a": [1]})

        dest = CSVDestination(output_path)
        dest.save(df)

        assert output_path.exists()

    def test_save_overwrites_existing(self, tmp_path):
        """CSVDestination.save() overwrites existing file."""
        output_path = tmp_path / "output.csv"

        # Create initial file
        df1 = pd.DataFrame({"a": [1]})
        CSVDestination(output_path).save(df1)

        # Overwrite with new data
        df2 = pd.DataFrame({"a": [2, 3, 4]})
        CSVDestination(output_path).save(df2)

        result = pd.read_csv(output_path)
        assert len(result) == 3


class TestExcelDestination:
    """Test ExcelDestination."""

    def test_save_dataframe(self, tmp_path):
        """ExcelDestination saves DataFrame to Excel."""
        path = tmp_path / "output.xlsx"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        dest = ExcelDestination(path)
        dest.save(df)

        assert path.exists()
        result = pd.read_excel(path)
        assert len(result) == 2

    def test_save_creates_directories(self, tmp_path):
        """ExcelDestination creates parent directories."""
        path = tmp_path / "nested" / "dir" / "output.xlsx"
        df = pd.DataFrame({"a": [1]})

        dest = ExcelDestination(path)
        dest.save(df)

        assert path.exists()


class TestJSONDestination:
    """Test JSONDestination."""

    def test_save_dataframe(self, tmp_path):
        """JSONDestination saves DataFrame to JSON."""
        path = tmp_path / "output.json"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        dest = JSONDestination(path)
        dest.save(df)

        assert path.exists()
        result = pd.read_json(path)
        assert len(result) == 2

    def test_save_pretty_print(self, tmp_path):
        """JSONDestination supports pretty printing."""
        path = tmp_path / "output.json"
        df = pd.DataFrame({"a": [1]})

        dest = JSONDestination(path, indent=4)
        dest.save(df)

        content = path.read_text()
        assert "    " in content  # 4-space indent


class TestDestinationsIntegration:
    """Integration tests for destinations."""

    def test_pipeline_with_excel(self, tmp_path):
        """Pipeline works with Excel destinations."""
        from analitica.etl import Pipeline, CSVSource, ExcelDestination

        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.xlsx"
        pd.DataFrame({"a": [1, 2]}).to_csv(input_path, index=False)

        pipeline = (
            Pipeline()
            .extract_from(CSVSource(input_path))
            .load_to(ExcelDestination(output_path))
        )
        result = pipeline.run()

        assert output_path.exists()

    def test_pipeline_with_json(self, tmp_path):
        """Pipeline works with JSON destinations."""
        from analitica.etl import Pipeline, CSVSource, JSONDestination

        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.json"
        pd.DataFrame({"a": [1, 2]}).to_csv(input_path, index=False)

        pipeline = (
            Pipeline()
            .extract_from(CSVSource(input_path))
            .load_to(JSONDestination(output_path))
        )
        result = pipeline.run()

        assert output_path.exists()
