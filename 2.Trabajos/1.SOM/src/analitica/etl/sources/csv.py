"""CSV data source."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..exceptions import SourceError
from .base import Source


class CSVSource(Source):
    """CSV file data source."""

    def __init__(self, path: str | Path, **kwargs):
        """Initialize CSV source.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional arguments passed to pandas.read_csv.
        """
        self.path = Path(path)
        self.kwargs = kwargs

    def extract(self) -> pd.DataFrame:
        """Extract data from CSV file.

        Returns:
            pd.DataFrame: The extracted data.

        Raises:
            SourceError: If file not found or parsing fails.
        """
        try:
            return pd.read_csv(self.path, **self.kwargs)
        except FileNotFoundError as e:
            raise SourceError(f"CSV file not found: {self.path}") from e
        except pd.errors.ParserError as e:
            raise SourceError(f"Failed to parse CSV: {e}") from e
        except Exception as e:
            raise SourceError(f"Failed to extract from CSV: {e}") from e
