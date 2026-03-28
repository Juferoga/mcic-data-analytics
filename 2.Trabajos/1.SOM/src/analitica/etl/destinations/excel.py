"""Excel data destination."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..exceptions import DestinationError
from .base import Destination


class ExcelDestination(Destination):
    """Excel file data destination (.xlsx)."""

    def __init__(self, path: str | Path, sheet_name: str = "Sheet1", **kwargs):
        """Initialize Excel destination.

        Args:
            path: Path to output Excel file.
            sheet_name: Sheet name. Default: 'Sheet1'.
            **kwargs: Additional arguments passed to pandas.to_excel.
        """
        self.path = Path(path)
        self.sheet_name = sheet_name
        self.kwargs = kwargs

    def save(self, data: pd.DataFrame) -> None:
        """Save data to Excel file.

        Args:
            data: The DataFrame to save.

        Raises:
            DestinationError: If writing fails.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data.to_excel(
                self.path, sheet_name=self.sheet_name, index=False, **self.kwargs
            )
        except PermissionError as e:
            raise DestinationError(f"Permission denied writing to: {self.path}") from e
        except Exception as e:
            raise DestinationError(f"Failed to save to Excel: {e}") from e
