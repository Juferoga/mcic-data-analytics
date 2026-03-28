"""JSON data destination."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..exceptions import DestinationError
from .base import Destination


class JSONDestination(Destination):
    """JSON file data destination."""

    def __init__(
        self,
        path: str | Path,
        orient: str = "records",
        indent: Optional[int] = 2,
        **kwargs,
    ):
        """Initialize JSON destination.

        Args:
            path: Path to output JSON file.
            orient: JSON orientation. Default: 'records'.
            indent: Pretty print indentation. Default: 2. Set to None for compact.
            **kwargs: Additional arguments passed to pandas.to_json.
        """
        self.path = Path(path)
        self.orient = orient
        self.indent = indent
        self.kwargs = kwargs

    def save(self, data: pd.DataFrame) -> None:
        """Save data to JSON file.

        Args:
            data: The DataFrame to save.

        Raises:
            DestinationError: If writing fails.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data.to_json(
                self.path, orient=self.orient, indent=self.indent, **self.kwargs
            )
        except PermissionError as e:
            raise DestinationError(f"Permission denied writing to: {self.path}") from e
        except Exception as e:
            raise DestinationError(f"Failed to save to JSON: {e}") from e
