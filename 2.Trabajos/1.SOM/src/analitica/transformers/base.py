"""Base encoder class for categorical transformers."""

from abc import abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from analitica.etl.transformer import BaseTransformer


class BaseEncoder(BaseTransformer):
    """Base class for categorical encoders.

    Provides common functionality for encoding categorical columns.
    Subclasses must implement _compute_mapping and _apply_encoding.
    """

    def __init__(
        self, columns: Optional[List[str]] = None, handle_unknown: str = "error"
    ):
        """Initialize encoder.

        Args:
            columns: List of columns to encode. If None, all object/category columns.
            handle_unknown: How to handle unseen categories. 'error', 'warn', or 'ignore'.
        """
        self.columns = columns
        self.handle_unknown = handle_unknown
        self._fitted_columns: List[str] = []
        self._mapping: Dict[str, Any] = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseEncoder":
        """Learn encoding mappings from data.

        Args:
            data: Input DataFrame.
            y: Target variable (required for supervised encoders).

        Returns:
            self: The fitted encoder.
        """
        columns = self.columns or self._get_categorical_columns(data)
        self._fitted_columns = [c for c in columns if c in data.columns]
        self._mapping = self._compute_mapping(data, y)
        return self

    @abstractmethod
    def _compute_mapping(
        self, data: pd.DataFrame, y: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Compute encoding mapping from data.

        Args:
            data: DataFrame with columns to encode.
            y: Target variable if supervised.

        Returns:
            Dict with encoding parameters.
        """
        pass

    @abstractmethod
    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the encoding transformation.

        Args:
            data: DataFrame with columns to encode.

        Returns:
            pd.DataFrame: Encoded data.
        """
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding transformation.

        Args:
            data: Input DataFrame.

        Returns:
            pd.DataFrame: Encoded data.
        """
        return self._apply_encoding(data)

    @staticmethod
    def _get_categorical_columns(data: pd.DataFrame) -> List[str]:
        """Get list of categorical columns."""
        return data.select_dtypes(include=["object", "category"]).columns.tolist()
