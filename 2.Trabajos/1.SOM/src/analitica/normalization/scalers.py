"""Scaler normalizers."""

from typing import Dict, Any

import numpy as np
import pandas as pd

from .base import BaseNormalizer


class MinMaxScaler(BaseNormalizer):
    """Scale features to [0, 1] range.

    Transform: x_scaled = (x - min) / (max - min)

    For constant columns (max == min), values are set to 0.
    """

    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute min and max for each column."""
        return {
            col: {"min": data[col].min(), "max": data[col].max()}
            for col in data.columns
        }

    def _apply_transform(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply Min-Max scaling."""
        result = data.copy()
        for col in data.columns:
            col_min = params[col]["min"]
            col_max = params[col]["max"]
            if col_max == col_min:
                # Constant column - set to 0
                result[col] = 0.0
            else:
                result[col] = (data[col] - col_min) / (col_max - col_min)
        return result


class ZScoreScaler(BaseNormalizer):
    """Standardize features to mean=0, std=1.

    Transform: x_scaled = (x - mean) / std

    For constant columns (std == 0), values are set to 0.
    """

    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute mean and std for each column."""
        return {
            col: {"mean": data[col].mean(), "std": data[col].std()}
            for col in data.columns
        }

    def _apply_transform(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply Z-Score scaling."""
        result = data.copy()
        for col in data.columns:
            col_mean = params[col]["mean"]
            col_std = params[col]["std"]
            if col_std == 0:
                # Constant column - set to 0
                result[col] = 0.0
            else:
                result[col] = (data[col] - col_mean) / col_std
        return result


class RobustScaler(BaseNormalizer):
    """Scale using median and interquartile range (IQR).

    Transform: x_scaled = (x - median) / IQR

    More resistant to outliers than MinMax and ZScore.
    For columns with IQR == 0, values are set to 0.
    """

    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute median and IQR for each column."""
        return {
            col: {
                "median": data[col].median(),
                "q1": data[col].quantile(0.25),
                "q3": data[col].quantile(0.75),
            }
            for col in data.columns
        }

    def _apply_transform(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply Robust scaling."""
        result = data.copy()
        for col in data.columns:
            median = params[col]["median"]
            q1 = params[col]["q1"]
            q3 = params[col]["q3"]
            iqr = q3 - q1
            if iqr == 0:
                # No spread - set to 0
                result[col] = 0.0
            else:
                result[col] = (data[col] - median) / iqr
        return result
