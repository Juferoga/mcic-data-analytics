"""Excel data source."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..exceptions import SourceError
from .base import Source


class ExcelSource(Source):
    """Excel file data source (.xlsx, .xls)."""

    def __init__(self, path: str | Path, sheet_name: Union[str, int] = 0, **kwargs):
        """Initialize Excel source.

        Args:
            path: Path to Excel file.
            sheet_name: Sheet name (str) or index (int). Default: 0 (first sheet).
            **kwargs: Additional arguments passed to pandas.read_excel.
        """
        self.path = Path(path)
        self.sheet_name = sheet_name
        self.kwargs = kwargs

    def extract(self) -> pd.DataFrame:
        """Extract data from Excel file.

        Returns:
            pd.DataFrame: The extracted data.

        Raises:
            SourceError: If file not found or parsing fails.
        """
        try:
            return pd.read_excel(self.path, sheet_name=self.sheet_name, **self.kwargs)
        except FileNotFoundError as e:
            raise SourceError(f"Excel file not found: {self.path}") from e
        except ValueError as e:
            raise SourceError(f"Invalid sheet name or data: {e}") from e
        except Exception as e:
            raise SourceError(f"Failed to extract from Excel: {e}") from e
