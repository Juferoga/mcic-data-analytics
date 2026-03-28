"""Tests for the ETL Pipeline."""

import pytest
import pandas as pd
from pathlib import Path

from analitica.etl import Pipeline, CSVSource, CSVDestination
from analitica.etl import IdentityTransformer, PipelineError
from analitica.etl.source import Source
from analitica.etl.destination import Destination
from analitica.etl.transformer import BaseTransformer


class TestPipeline:
    """Test cases for Pipeline class."""

    def test_fluent_interface_extract_from(self, tmp_path):
        """Pipeline.extract_from() returns self for chaining."""
        csv_path = tmp_path / "input.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        source = CSVSource(csv_path)
        pipeline = Pipeline().extract_from(source)

        assert pipeline is not None

    def test_fluent_interface_add_transformer(self):
        """Pipeline.add_transformer() returns self for chaining."""
        pipeline = Pipeline()
        transformer = IdentityTransformer()
        result = pipeline.add_transformer(transformer)

        assert result is pipeline

    def test_fluent_interface_load_to(self):
        """Pipeline.load_to() returns self for chaining."""
        pipeline = Pipeline()
        # Create a mock destination
        dest = type("MockDestination", (Destination,), {"save": lambda s, d: None})()
        result = pipeline.load_to(dest)

        assert result is pipeline

    def test_pipeline_execution_success(self, tmp_path):
        """Pipeline.run() executes successfully with source and destination."""
        # Create input CSV
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}).to_csv(
            input_path, index=False
        )

        # Run pipeline
        pipeline = (
            Pipeline()
            .extract_from(CSVSource(input_path))
            .add_transformer(IdentityTransformer())
            .load_to(CSVDestination(output_path))
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert output_path.exists()

    def test_pipeline_execution_no_source(self):
        """Pipeline.run() raises PipelineError without source."""
        pipeline = Pipeline()

        with pytest.raises(PipelineError, match="No source configured"):
            pipeline.run()

    def test_pipeline_with_multiple_transformers(self, tmp_path):
        """Pipeline applies multiple transformers in order."""
        input_path = tmp_path / "input.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(input_path, index=False)

        pipeline = (
            Pipeline()
            .extract_from(CSVSource(input_path))
            .add_transformer(IdentityTransformer())
            .add_transformer(IdentityTransformer())
        )
        result = pipeline.run()

        assert len(result) == 3
