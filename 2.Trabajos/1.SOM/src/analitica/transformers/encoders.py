"""Categorical encoders for text-to-number transformation."""

import hashlib
import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .base import BaseEncoder


class LabelEncoder(BaseEncoder):
    """Encode categorical labels as integers.

    Preserves ordinal relationship between categories.
    Suitable for ordinal data (low/medium/high, small/medium/large).

    Example:
        ['cat', 'dog', 'bird'] -> [0, 1, 2]
    """

    def _compute_mapping(
        self, data: pd.DataFrame, y: Optional[pd.Series]
    ) -> Dict[str, Dict[str, int]]:
        """Compute label to integer mapping."""
        mapping = {}
        for col in self._fitted_columns:
            unique_values = data[col].unique()
            mapping[col] = {val: idx for idx, val in enumerate(sorted(unique_values))}
        return mapping

    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding."""
        result = data.copy()
        for col in self._fitted_columns:
            if col not in self._mapping:
                continue

            mapping = self._mapping[col]
            known_categories = set(mapping.keys())
            values = data[col].values

            # Check for unknown categories
            if self.handle_unknown != "ignore":
                unknown_mask = ~pd.Series(values).isin(known_categories)
                if unknown_mask.any():
                    if self.handle_unknown == "error":
                        unknown_vals = set(pd.Series(values)[unknown_mask].unique())
                        raise ValueError(
                            f"Unknown categories in '{col}': {unknown_vals}"
                        )
                    else:
                        warnings.warn(
                            f"Unknown categories found in '{col}', encoding as -1"
                        )

            # Apply encoding
            encoded = pd.Series(values).map(mapping)
            if self.handle_unknown != "ignore":
                encoded = encoded.fillna(-1).astype(int)
            result[col] = encoded

        return result


class OneHotEncoder(BaseEncoder):
    """One-hot encode categorical variables.

    Creates binary columns for each category.
    Suitable for nominal data (city, color, country).

    Example:
        'city': ['NY', 'LA', 'NY'] -> 'city_NY': [1, 0, 1], 'city_LA': [0, 1, 0]
    """

    def __init__(
        self,
        columns: Optional[list] = None,
        drop_first: bool = False,
        handle_unknown: str = "ignore",
        prefix_sep: str = "_",
    ):
        """Initialize one-hot encoder.

        Args:
            columns: Columns to encode.
            drop_first: Drop first category to avoid multicollinearity.
            handle_unknown: How to handle unknown categories.
            prefix_sep: Separator between column name and category.
        """
        super().__init__(columns, handle_unknown)
        self.drop_first = drop_first
        self.prefix_sep = prefix_sep

    def _compute_mapping(
        self, data: pd.DataFrame, y: Optional[pd.Series]
    ) -> Dict[str, list]:
        """Compute list of categories per column."""
        mapping = {}
        for col in self._fitted_columns:
            categories = sorted(data[col].unique().tolist())
            if self.drop_first:
                categories = categories[1:]
            mapping[col] = categories
        return mapping

    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding."""
        result = data.copy()

        for col in self._fitted_columns:
            if col not in self._mapping:
                continue

            categories = self._mapping[col]
            prefix = col

            # Create binary columns
            for category in categories:
                new_col = f"{prefix}{self.prefix_sep}{category}"
                result[new_col] = (data[col] == category).astype(int)

            # Drop original column
            result = result.drop(columns=[col])

        return result


class TargetEncoder(BaseEncoder):
    """Encode categorical variables using target variable mean.

    Supervised encoding that replaces categories with their mean target value.
    Includes smoothing to handle rare categories.

    Formula: encoded = (count * category_mean + smoothing * global_mean) / (count + smoothing)

    Example:
        'city': ['NY', 'LA', 'SF'] with target 'purchase': [1,0,1]
        -> NY_mean=0.8, LA_mean=0.2, SF_mean=0.9
    """

    def __init__(
        self,
        columns: Optional[list] = None,
        smoothing: float = 1.0,
        handle_unknown: str = "ignore",
    ):
        """Initialize target encoder.

        Args:
            columns: Columns to encode.
            smoothing: Smoothing factor (higher = more regularization).
            handle_unknown: How to handle unknown categories.
        """
        super().__init__(columns, handle_unknown)
        self.smoothing = smoothing
        self._global_mean: float = 0.0

    def fit(self, data: pd.DataFrame, y: pd.Series = None) -> "TargetEncoder":
        """Learn target encoding mappings.

        Args:
            data: Input DataFrame.
            y: Target variable (required).

        Returns:
            self: The fitted encoder.
        """
        if y is None:
            raise ValueError("TargetEncoder requires target variable 'y'")

        if len(data) != len(y):
            raise ValueError("Data and target must have same length")

        columns = self.columns or self._get_categorical_columns(data)
        self._fitted_columns = [c for c in columns if c in data.columns]
        self._global_mean = y.mean()
        self._mapping = self._compute_mapping(data, y)
        return self

    def fit_transform(self, data: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform data.

        Args:
            data: Input DataFrame.
            y: Target variable (required).

        Returns:
            pd.DataFrame: Encoded data.
        """
        return self.fit(data, y).transform(data)

    def _compute_mapping(
        self, data: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean target per category."""
        mapping = {}
        for col in self._fitted_columns:
            df_temp = pd.DataFrame({col: data[col], "target": y})
            stats = df_temp.groupby(col)["target"].agg(["mean", "count"])

            # Apply smoothing
            global_mean = self._global_mean
            smoothed_means = (
                stats["count"] * stats["mean"] + self.smoothing * global_mean
            ) / (stats["count"] + self.smoothing)
            mapping[col] = smoothed_means.to_dict()
        return mapping

    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding."""
        result = data.copy()
        for col in self._fitted_columns:
            if col not in self._mapping:
                continue

            mapping = self._mapping[col]
            result[col] = data[col].map(mapping)

            # Handle unknown categories
            if self.handle_unknown != "ignore":
                unknown_mask = result[col].isna()
                if unknown_mask.any():
                    if self.handle_unknown == "error":
                        unknown_vals = data.loc[unknown_mask, col].unique()
                        raise ValueError(
                            f"Unknown categories in '{col}': {unknown_vals}"
                        )
                    else:
                        warnings.warn(
                            f"Unknown categories in '{col}', using global mean"
                        )
                        result.loc[unknown_mask, col] = self._global_mean
            else:
                result[col] = result[col].fillna(self._global_mean)

            result[col] = result[col].astype(float)

        return result


class HashEncoder(BaseEncoder):
    """Hash categorical variables to a fixed number of bins.

    Memory-efficient encoding for high-cardinality features.
    Uses multiple hash functions to reduce collisions.

    Example:
        'url': ['http://a.com', 'http://b.com', 'http://c.com']
        with n_bins=2 -> 'url_hash_0': [1, 0, 1], 'url_hash_1': [0, 1, 0]
    """

    def __init__(
        self,
        columns: Optional[list] = None,
        n_bins: int = 8,
        n_functions: int = 3,
        handle_unknown: str = "ignore",
    ):
        """Initialize hash encoder.

        Args:
            columns: Columns to encode.
            n_bins: Number of hash buckets.
            n_functions: Number of independent hash functions.
            handle_unknown: How to handle unknown categories.
        """
        super().__init__(columns, handle_unknown)
        self.n_bins = n_bins
        self.n_functions = n_functions

    def _compute_mapping(
        self, data: pd.DataFrame, y: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Store encoder configuration (stateless)."""
        return {"n_bins": self.n_bins, "n_functions": self.n_functions}

    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply hash encoding."""
        result = data.copy()

        for col in self._fitted_columns:
            prefix = col

            for i in range(self.n_functions):
                # Create hash function with different seed
                def hash_func(val, idx=i):
                    if pd.isna(val):
                        return 0
                    h = hashlib.md5(f"{col}_{idx}_{val}".encode()).hexdigest()
                    return int(h, 16) % self.n_bins

                new_col = f"{prefix}_hash_{i}"
                result[new_col] = data[col].apply(lambda x: hash_func(x, i))

            # Drop original column
            result = result.drop(columns=[col])

        return result
