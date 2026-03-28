"""ETL Pipeline orchestration."""

from typing import Optional, List

import pandas as pd

from .source import Source
from .destination import Destination
from .transformer import BaseTransformer
from .exceptions import PipelineError


class Pipeline:
    """ETL Pipeline with fluent interface."""

    def __init__(
        self,
        source: Optional[Source] = None,
        destination: Optional[Destination] = None,
        transformers: Optional[List[BaseTransformer]] = None,
    ):
        """Initialize pipeline.

        Args:
            source: Data source.
            destination: Data destination.
            transformers: List of transformers to apply.
        """
        self._source = source
        self._destination = destination
        self._transformers: List[BaseTransformer] = transformers or []
        self._data: Optional[pd.DataFrame] = None

    def extract_from(self, source: Source) -> "Pipeline":
        """Set the data source.

        Args:
            source: The data source.

        Returns:
            Pipeline: self for chaining.
        """
        self._source = source
        return self

    def add_transformer(self, transformer: BaseTransformer) -> "Pipeline":
        """Add a transformer to the pipeline.

        Args:
            transformer: The transformer to add.

        Returns:
            Pipeline: self for chaining.
        """
        self._transformers.append(transformer)
        return self

    def load_to(self, destination: Destination) -> "Pipeline":
        """Set the data destination.

        Args:
            destination: The data destination.

        Returns:
            Pipeline: self for chaining.
        """
        self._destination = destination
        return self

    def run(self) -> pd.DataFrame:
        """Execute the pipeline.

        Returns:
            pd.DataFrame: The transformed data.

        Raises:
            PipelineError: If pipeline execution fails.
        """
        try:
            # Extract
            if self._source is None:
                raise PipelineError("No source configured. Call extract_from() first.")
            self._data = self._source.extract()

            # Transform
            for transformer in self._transformers:
                self._data = transformer.fit_transform(self._data)

            # Load
            if self._destination is not None:
                self._destination.save(self._data)

            return self._data

        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Pipeline execution failed: {e}") from e

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the current data state.

        Returns:
            pd.DataFrame or None: The current data.
        """
        return self._data
