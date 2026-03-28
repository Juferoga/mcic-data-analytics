"""Abstract base class for data destinations."""

from abc import ABC, abstractmethod

import pandas as pd


class Destination(ABC):
    """Abstract base class for data destinations."""

    @abstractmethod
    def save(self, data: pd.DataFrame) -> None:
        """Save data to the destination.

        Args:
            data: The DataFrame to save.

        Raises:
            DestinationError: If saving fails.
        """
        pass
