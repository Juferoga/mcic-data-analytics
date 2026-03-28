"""JSON data source."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..exceptions import SourceError
from .base import Source


class JSONSource(Source):
    """JSON file data source."""

    def __init__(self, path: str | Path, orient: str = "records", **kwargs):
        """Initialize JSON source.

        Args:
            path: Path to JSON file.
            orient: JSON orientation. Default: 'records' (array of objects).
            **kwargs: Additional arguments passed to pandas.read_json.
        """
        self.path = Path(path)
        self.orient = orient
        self.kwargs = kwargs

    def extract(self) -> pd.DataFrame:
        """Extract data from JSON file.

        Returns:
            pd.DataFrame: The extracted data.

        Raises:
            SourceError: If file not found or parsing fails.
        """
        try:
            return pd.read_json(self.path, orient=self.orient, **self.kwargs)
        except FileNotFoundError as e:
            raise SourceError(f"JSON file not found: {self.path}") from e
        except ValueError as e:
            raise SourceError(f"Invalid JSON format: {e}") from e
        except Exception as e:
            raise SourceError(f"Failed to extract from JSON: {e}") from e
