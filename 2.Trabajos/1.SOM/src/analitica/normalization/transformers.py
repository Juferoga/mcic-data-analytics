"""Non-linear transformers."""

from typing import Dict, Any

import numpy as np
import pandas as pd

from .base import BaseNormalizer


class LogTransformer(BaseNormalizer):
    """Apply log transformation to reduce skewness.

    Automatically handles negative values by shifting: x_shifted = x + shift
    where shift = |min| + 1 if min <= 0, else 0.

    Transform: x_transformed = log(x_shifted)
    """

    def __init__(self, columns=None, base: str = "natural"):
        """Initialize log transformer.

        Args:
            columns: Columns to transform.
            base: 'natural' for ln, or '10' for log10.
        """
        super().__init__(columns)
        self.base = base

    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute shift value for each column."""
        params = {}
        for col in data.columns:
            col_min = data[col].min()
            if col_min <= 0:
                shift = abs(col_min) + 1
            else:
                shift = 0
            params[col] = {"shift": shift}
        return params

    def _apply_transform(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply log transformation."""
        result = data.copy()
        for col in data.columns:
            shift = params[col]["shift"]
            x = data[col] + shift
            if self.base == "10":
                result[col] = np.log10(x)
            else:
                result[col] = np.log(x)
        return result


class PowerTransformer(BaseNormalizer):
    """Apply Yeo-Johnson power transformation.

    Makes data more Gaussian-like, useful for stabilizing variance.
    Uses sklearn.preprocessing.PowerTransformer.
    """

    def __init__(self, columns=None, method: str = "yeo-johnson"):
        """Initialize power transformer.

        Args:
            columns: Columns to transform.
            method: 'yeo-johnson' or 'box-cox'.
        """
        super().__init__(columns)
        self.method = method
        self._pt = None

    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Learn power transformation parameters."""
        from sklearn.preprocessing import PowerTransformer

        self._pt = PowerTransformer(method=self.method, standardize=True)
        self._pt.fit(data.values)

        return {"n_features": data.shape[1]}

    def _apply_transform(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply power transformation."""
        result = data.copy()
        if self._pt is not None:
            transformed = self._pt.transform(data.values)
            for i, col in enumerate(data.columns):
                result[col] = transformed[:, i]
        return result
