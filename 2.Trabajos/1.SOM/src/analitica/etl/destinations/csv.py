"""CSV data destination."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..exceptions import DestinationError
from .base import Destination


class CSVDestination(Destination):
    """CSV file data destination."""

    def __init__(self, path: str | Path, **kwargs):
        """Initialize CSV destination.

        Args:
            path: Path to the output CSV file.
            **kwargs: Additional arguments passed to pandas.to_csv.
        """
        self.path = Path(path)
        self.kwargs = kwargs

    def save(self, data: pd.DataFrame) -> None:
        """Save data to CSV file.

        Args:
            data: The DataFrame to save.

        Raises:
            DestinationError: If writing fails.
        """
        try:
            # Create parent directories if they don't exist
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(self.path, index=False, **self.kwargs)
        except PermissionError as e:
            raise DestinationError(f"Permission denied writing to: {self.path}") from e
        except Exception as e:
            raise DestinationError(f"Failed to save to CSV: {e}") from e
