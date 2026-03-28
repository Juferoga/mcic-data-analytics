"""Base normalizer class."""

from abc import abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from analitica.etl.transformer import BaseTransformer


class BaseNormalizer(BaseTransformer):
    """Base class for all normalizers.

    Provides common functionality for column selection and parameter storage.
    Subclasses must implement _compute_params and _apply_transform.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize normalizer.

        Args:
            columns: List of columns to normalize. If None, all numeric columns.
        """
        self.columns = columns
        self._fitted_columns: List[str] = []
        self._params: Dict[str, Any] = {}

    def fit(self, data: pd.DataFrame) -> "BaseNormalizer":
        """Learn normalization parameters from data.

        Args:
            data: Input DataFrame.

        Returns:
            self: The fitted normalizer.
        """
        columns = self.columns or self._get_numeric_columns(data)
        self._fitted_columns = [c for c in columns if c in data.columns]
        self._params = self._compute_params(data[self._fitted_columns])
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization.

        Args:
            data: Input DataFrame.

        Returns:
            pd.DataFrame: Normalized data.
        """
        result = data.copy()
        if self._fitted_columns:
            result[self._fitted_columns] = self._apply_transform(
                data[self._fitted_columns], self._params
            )
        return result

    @abstractmethod
    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute normalization parameters from data.

        Args:
            data: DataFrame with columns to normalize.

        Returns:
            Dict with parameters needed for transformation.
        """
        pass

    @abstractmethod
    def _apply_transform(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply the normalization transformation.

        Args:
            data: DataFrame with columns to transform.
            params: Parameters from fit.

        Returns:
            pd.DataFrame: Transformed data.
        """
        pass

    @staticmethod
    def _get_numeric_columns(data: pd.DataFrame) -> List[str]:
        """Get list of numeric columns."""
        return data.select_dtypes(include=[np.number]).columns.tolist()
