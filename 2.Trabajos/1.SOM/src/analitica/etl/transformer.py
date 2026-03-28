"""Data transformation components."""

from abc import ABC, abstractmethod

import pandas as pd

from .exceptions import TransformerError


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""

    def fit(self, data: pd.DataFrame) -> "BaseTransformer":
        """Learn parameters from data.

        Args:
            data: The input DataFrame.

        Returns:
            self: The fitted transformer.
        """
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data.

        Args:
            data: The input DataFrame.

        Returns:
            pd.DataFrame: The transformed data.

        Raises:
            TransformerError: If transformation fails.
        """
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data.

        Args:
            data: The input DataFrame.

        Returns:
            pd.DataFrame: The transformed data.
        """
        return self.fit(data).transform(data)


class IdentityTransformer(BaseTransformer):
    """No-op transformer that returns data unchanged."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return data unchanged.

        Args:
            data: The input DataFrame.

        Returns:
            pd.DataFrame: The same data.
        """
        return data
