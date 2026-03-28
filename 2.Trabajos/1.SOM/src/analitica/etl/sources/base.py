"""Abstract base class for data sources."""

from abc import ABC, abstractmethod

import pandas as pd


class Source(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data from the source.

        Returns:
            pd.DataFrame: The extracted data.

        Raises:
            SourceError: If extraction fails.
        """
        pass
