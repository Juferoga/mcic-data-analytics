# Technical Design: Text-to-Number Encoders

## Overview

This document describes the technical design for implementing text/categorical encoders in the `analitica` library. The encoders transform categorical and text data into numerical representations required by SOM and clustering algorithms.

## Architecture

### Class Hierarchy

```
BaseTransformer (from etl/transformer.py)
    │
    └── BaseEncoder (transformers/base.py)
            │
            ├── LabelEncoder (transformers/encoders.py)
            ├── OneHotEncoder (transformers/encoders.py)
            ├── TargetEncoder (transformers/encoders.py)
            └── HashEncoder (transformers/encoders.py)
```

### Design Patterns

1. **sklearn-compatible API**: All encoders follow fit/transform/fit_transform pattern
2. **Immutable fit parameters**: Mappings learned during `fit()` stored and reused in `transform()`
3. **Column selection**: Supports explicit column specification via `columns` parameter
4. **Unseen category handling**: Log warning and encode to default value

---

## File Structure

```
src/analitica/transformers/
├── __init__.py          # Modified: Export all encoders
├── base.py              # New: BaseEncoder abstract class
└── encoders.py         # New: All 4 encoder implementations
```

---

## Data Flow

### Encoder Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                       fit(data)                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Select columns (explicit or all object columns)│   │
│  │ 2. Compute encoding mapping                         │   │
│  │ 3. Store in self._mapping                           │   │
│  │ 4. Return self                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    transform(data)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Copy input DataFrame                             │   │
│  │ 2. Apply encoding using stored mapping              │   │
│  │ 3. Handle unseen categories (warn + default)         │   │
│  │ 4. Return transformed DataFrame                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Integration

```python
from analitica.etl import Pipeline
from analitica.transformers import LabelEncoder, OneHotEncoder

pipeline = Pipeline()
pipeline.add_source(CSVSource("data.csv"))
pipeline.add_transformer(LabelEncoder(columns=["category"]))
pipeline.add_transformer(OneHotEncoder(columns=["city"]))
pipeline.execute()
```

---

## Implementation Details

### 1. BaseEncoder (base.py)

```python
"""Base encoder class."""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
import warnings

import pandas as pd

from analitica.etl.transformer import BaseTransformer


class BaseEncoder(BaseTransformer):
    """Base class for all encoders.
    
    Provides common functionality for categorical column selection
    and mapping storage. Subclasses implement _compute_mapping and
    _apply_encoding.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize encoder.
        
        Args:
            columns: List of columns to encode. If None, all object/category
                    columns are processed.
        """
        self.columns = columns
        self._fitted_columns: List[str] = []
        self._mapping: Dict[str, Any] = {}
    
    def fit(self, data: pd.DataFrame) -> "BaseEncoder":
        """Learn encoding mapping from data.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            self: The fitted encoder.
        """
        columns = self.columns or self._get_categorical_columns(data)
        self._fitted_columns = [c for c in columns if c in data.columns]
        self._mapping = self._compute_mapping(data[self._fitted_columns])
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to data.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            pd.DataFrame: Encoded data.
            
        Raises:
            RuntimeError: If encoder has not been fitted.
        """
        if not self._fitted_columns:
            raise RuntimeError("Encoder has not been fitted. Call fit() first.")
        
        result = data.copy()
        for col in self._fitted_columns:
            if col in result.columns:
                result[col] = self._apply_encoding(
                    result[col], self._mapping.get(col, {})
                )
        return result
    
    @abstractmethod
    def _compute_mapping(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute encoding mapping from data.
        
        Args:
            data: DataFrame with columns to encode.
            
        Returns:
            Dict mapping column names to encoding rules.
        """
        pass
    
    @abstractmethod
    def _apply_encoding(
        self, series: pd.Series, mapping: Any
    ) -> pd.Series:
        """Apply encoding to a single column.
        
        Args:
            series: Series to encode.
            mapping: Encoding mapping from fit().
            
        Returns:
            Encoded Series.
        """
        pass
    
    @staticmethod
    def _get_categorical_columns(data: pd.DataFrame) -> List[str]:
        """Get list of categorical (object/category) columns."""
        return data.select_dtypes(include=["object", "category"]).columns.tolist()
```

### 2. Encoders (encoders.py)

```python
"""Encoder implementations for categorical data."""

import hashlib
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd

from .base import BaseEncoder


class LabelEncoder(BaseEncoder):
    """Encode ordinal categories as integers.
    
    Maps each unique category to a sequential integer.
    Preserves ordinal relationships when categories have natural ordering.
    """
    
    def _compute_mapping(self, data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Compute category to integer mapping.
        
        Args:
            data: DataFrame with columns to encode.
            
        Returns:
            Dict mapping column names to {category: integer} dicts.
        """
        mapping = {}
        for col in data.columns:
            unique_vals = data[col].dropna().unique()
            mapping[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return mapping
    
    def _apply_encoding(
        self, series: pd.Series, mapping: Dict[str, int]
    ) -> pd.Series:
        """Apply label encoding.
        
        Args:
            series: Series to encode.
            mapping: Category to integer mapping.
            
        Returns:
            Encoded Series with -1 for unseen values.
        """
        default = -1  # For unseen categories
        
        # Check for unseen categories
        unseen = ~series.isin(mapping.keys()) & series.notna()
        if unseen.any():
            warnings.warn(
                f"Found {unseen.sum()} unseen categories in {series.name}. "
                f"These will be encoded as {default}.",
                UserWarning
            )
        
        return series.map(mapping).fillna(default).astype(int)


class OneHotEncoder(BaseEncoder):
    """Encode nominal categories as binary columns.
    
    Creates new columns for each unique category (one column per category).
    Suitable for nominal categories without natural ordering.
    """
    
    def __init__(self, columns: List[str] = None, drop_first: bool = False):
        """Initialize OneHotEncoder.
        
        Args:
            columns: List of columns to encode.
            drop_first: Whether to drop first category to avoid multicollinearity.
        """
        super().__init__(columns)
        self.drop_first = drop_first
    
    def _compute_mapping(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Compute unique categories per column.
        
        Args:
            data: DataFrame with columns to encode.
            
        Returns:
            Dict mapping column names to sorted list of categories.
        """
        mapping = {}
        for col in data.columns:
            categories = sorted(data[col].dropna().unique().tolist())
            if self.drop_first and categories:
                categories = categories[1:]
            mapping[col] = categories
        return mapping
    
    def _apply_encoding(
        self, series: pd.Series, categories: List[str]
    ) -> pd.DataFrame:
        """Apply one-hot encoding.
        
        Args:
            series: Series to encode.
            categories: List of valid categories.
            
        Returns:
            DataFrame with one column per category.
        """
        result = pd.DataFrame(index=series.index)
        for cat in categories:
            col_name = f"{series.name}_{cat}"
            result[col_name] = (series == cat).astype(int)
        return result
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding (returns DataFrame, not Series per column)."""
        if not self._fitted_columns:
            raise RuntimeError("Encoder has not been fitted. Call fit() first.")
        
        result = data.copy()
        # Drop original columns and add one-hot columns
        for col in self._fitted_columns:
            if col in result.columns:
                encoded = self._apply_encoding(
                    result[col], self._mapping.get(col, [])
                )
                result = result.drop(columns=[col])
                result = pd.concat([result, encoded], axis=1)
        return result


class TargetEncoder(BaseEncoder):
    """Encode categories using target variable mean (supervised).
    
    Replaces each category with the mean of the target variable for that
    category. Uses smoothing to handle rare categories.
    """
    
    def __init__(
        self,
        columns: List[str],
        target: str,
        smoothing: float = 1.0,
        min_samples: int = 1
    ):
        """Initialize TargetEncoder.
        
        Args:
            columns: List of columns to encode (required).
            target: Name of target column for computing means.
            smoothing: Smoothing factor (higher = more regularization).
            min_samples: Minimum samples for category to use its own mean.
        """
        super().__init__(columns)
        self.target = target
        self.smoothing = smoothing
        self.min_samples = min_samples
        self._global_mean: float = 0.0
    
    def fit(self, data: pd.DataFrame) -> "TargetEncoder":
        """Learn target mean encoding from data.
        
        Args:
            data: Input DataFrame with target column.
            
        Returns:
            self: The fitted encoder.
            
        Raises:
            ValueError: If target column not in data.
        """
        if self.target not in data.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        
        columns = self.columns or self._get_categorical_columns(data)
        self._fitted_columns = [c for c in columns if c in data.columns]
        self._global_mean = data[self.target].mean()
        self._mapping = self._compute_mapping(data, self.target)
        return self
    
    def _compute_mapping(
        self, data: pd.DataFrame, target: str
    ) -> Dict[str, Dict[str, float]]:
        """Compute smoothed target mean per category.
        
        Formula: encoded = (count * mean + smoothing * global_mean) / (count + smoothing)
        
        Args:
            data: DataFrame with columns and target.
            target: Target column name.
            
        Returns:
            Dict mapping column names to {category: encoded_value} dicts.
        """
        mapping = {}
        global_mean = data[target].mean()
        
        for col in self._fitted_columns:
            col_mapping = {}
            for cat, group in data.groupby(col)[target]:
                count = len(group)
                mean = group.mean()
                
                if count < self.min_samples:
                    col_mapping[cat] = global_mean
                else:
                    # Smoothing formula
                    col_mapping[cat] = (
                        (count * mean + self.smoothing * global_mean) /
                        (count + self.smoothing)
                    )
            mapping[col] = col_mapping
        
        return mapping
    
    def _apply_encoding(
        self, series: pd.Series, mapping: Dict[str, float]
    ) -> pd.Series:
        """Apply target encoding.
        
        Args:
            series: Series to encode.
            mapping: Category to mean mapping.
            
        Returns:
            Encoded Series with global mean for unseen categories.
        """
        return series.map(mapping).fillna(self._global_mean)


class HashEncoder(BaseEncoder):
    """Encode categories using hash function (for high-cardinality).
    
    Uses consistent hashing to map any category to a fixed number of buckets.
    Memory-efficient for very high cardinality features (URLs, IDs).
    """
    
    def __init__(
        self,
        columns: List[str],
        n_bins: int = 32,
        signed: bool = False
    ):
        """Initialize HashEncoder.
        
        Args:
            columns: List of columns to encode.
            n_bins: Number of hash buckets (output range: 0 to n_bins-1).
            signed: If True, output range is -n_bins/2 to n_bins/2.
        """
        super().__init__(columns)
        self.n_bins = n_bins
        self.signed = signed
    
    def _compute_mapping(self, data: pd.DataFrame) -> Dict[str, None]:
        """Hash encoding uses no mapping (stateless).
        
        Args:
            data: DataFrame with columns to encode.
            
        Returns:
            Empty dict (transformation is stateless).
        """
        return {col: None for col in data.columns}
    
    def _apply_encoding(
        self, series: pd.Series, mapping: Any
    ) -> pd.Series:
        """Apply hash encoding.
        
        Args:
            series: Series to encode.
            mapping: Not used (stateless).
            
        Returns:
            Encoded Series with hashed integer values.
        """
        def hash_value(val):
            if pd.isna(val):
                return 0
            # Use MD5 hash, take first 8 bytes as integer
            h = hashlib.md5(str(val).encode()).hexdigest()[:8]
            hashed = int(h, 16) % self.n_bins
            if self.signed:
                hashed = hashed - self.n_bins // 2
            return hashed
        
        return series.apply(hash_value)
```

### 3. Module Exports (__init__.py)

```python
"""Transformers Module - Encoders for categorical/text data."""

from .base import BaseEncoder
from .encoders import (
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder,
    HashEncoder,
)

__all__ = [
    "BaseEncoder",
    "LabelEncoder",
    "OneHotEncoder",
    "TargetEncoder",
    "HashEncoder",
]
```

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| No categorical columns | Raise `ValueError` with descriptive message |
| Not fitted before transform | Raise `RuntimeError` |
| Unseen categories | Log warning, encode as default (-1 for LabelEncoder) |
| TargetEncoder missing target column | Raise `ValueError` |
| Empty DataFrame | Allow through (returns empty DataFrame) |

---

## Testing Strategy

### Test File Structure

```
tests/
├── test_encoders.py    # Main test file
```

### Test Categories

#### 1. Unit Tests for BaseEncoder

```python
# tests/test_encoders.py
import pytest
import pandas as pd
import numpy as np
import warnings
from analitica.transformers import (
    BaseEncoder, LabelEncoder, OneHotEncoder, TargetEncoder, HashEncoder
)

class TestBaseEncoder:
    def test_fit_stores_columns(self):
        encoder = LabelEncoder()
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["a", "b", "c"]})
        
        encoder.fit(df)
        
        assert encoder._fitted_columns == ["a", "b"]
    
    def test_fit_with_explicit_columns(self):
        encoder = LabelEncoder(columns=["a"])
        df = pd.DataFrame({"a": ["x", "y"], "b": ["a", "b"]})
        
        encoder.fit(df)
        
        assert encoder._fitted_columns == ["a"]
    
    def test_transform_raises_if_not_fitted(self):
        encoder = LabelEncoder()
        df = pd.DataFrame({"a": ["x", "y"]})
        
        with pytest.raises(RuntimeError):
            encoder.transform(df)
```

#### 2. LabelEncoder Tests

```python
class TestLabelEncoder:
    def test_encodes_unique_values(self):
        encoder = LabelEncoder()
        df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b"]})
        
        result = encoder.fit_transform(df)
        
        assert set(result["cat"].unique()) == {0, 1, 2}
    
    def test_warns_on_unseen_categories(self):
        encoder = LabelEncoder()
        train_df = pd.DataFrame({"cat": ["a", "b", "c"]})
        test_df = pd.DataFrame({"cat": ["a", "d"]})  # 'd' is unseen
        
        encoder.fit(train_df)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = encoder.transform(test_df)
            assert len(w) == 1
            assert "unseen" in str(w[0].message).lower()
    
    def test_preserves_order_for_ordinal(self):
        encoder = LabelEncoder()
        df = pd.DataFrame({"level": ["low", "medium", "high"]})
        
        result = encoder.fit_transform(df)
        
        assert result["level"].tolist() == [0, 1, 2]
```

#### 3. OneHotEncoder Tests

```python
class TestOneHotEncoder:
    def test_creates_binary_columns(self):
        encoder = OneHotEncoder()
        df = pd.DataFrame({"city": ["NYC", "LA", "NYC"]})
        
        result = encoder.fit_transform(df)
        
        assert "city_NYC" in result.columns
        assert "city_LA" in result.columns
        assert result["city_NYC"].sum() == 2
    
    def test_drop_first(self):
        encoder = OneHotEncoder(drop_first=True)
        df = pd.DataFrame({"city": ["NYC", "LA", "SF"]})
        
        result = encoder.fit_transform(df)
        
        assert "city_LA" in result.columns
        assert "city_SF" in result.columns
        assert "city_NYC" not in result.columns
    
    def test_drops_original_column(self):
        encoder = OneHotEncoder()
        df = pd.DataFrame({"cat": ["a", "b"]})
        
        result = encoder.fit_transform(df)
        
        assert "cat" not in result.columns
```

#### 4. TargetEncoder Tests

```python
class TestTargetEncoder:
    def test_requires_target_column(self):
        encoder = TargetEncoder(columns=["cat"], target="missing")
        df = pd.DataFrame({"cat": ["a", "b"], "other": [1, 2]})
        
        with pytest.raises(ValueError) as exc:
            encoder.fit(df)
        assert "target" in str(exc.value).lower()
    
    def test_encodes_with_target_mean(self):
        encoder = TargetEncoder(columns=["cat"], target="value", smoothing=1.0)
        df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "value": [10, 20, 100, 200]
        })
        
        result = encoder.fit_transform(df)
        
        # a's mean = 15, b's mean = 150
        assert result["cat"].iloc[0] == pytest.approx(15, rel=0.1)
        assert result["cat"].iloc[2] == pytest.approx(150, rel=0.1)
    
    def test_smoothing_for_rare_categories(self):
        encoder = TargetEncoder(columns=["cat"], target="value", smoothing=10)
        df = pd.DataFrame({
            "cat": ["a", "a", "b"],  # b appears once
            "value": [10, 20, 100]
        })
        
        result = encoder.fit_transform(df)
        global_mean = 130 / 3
        
        # b should be smoothed toward global mean
        assert result["cat"].iloc[2] < 100  # Less than raw mean
```

#### 5. HashEncoder Tests

```python
class TestHashEncoder:
    def test_consistent_hashing(self):
        encoder = HashEncoder(columns=["url"], n_bins=100)
        df = pd.DataFrame({"url": ["https://example.com"] * 3})
        
        result = encoder.fit_transform(df)
        
        # Same input should always produce same output
        assert result["url"].nunique() == 1
    
    def test_deterministic_different_inputs(self):
        encoder = HashEncoder(columns=["url"], n_bins=100)
        df = pd.DataFrame({"url": ["https://a.com", "https://b.com"]})
        
        result = encoder.fit_transform(df)
        
        assert result["url"].iloc[0] != result["url"].iloc[1]
    
    def test_bounded_output(self):
        encoder = HashEncoder(columns=["val"], n_bins=32, signed=False)
        df = pd.DataFrame({"val": ["a", "b", "c", "d", "e"]})
        
        result = encoder.fit_transform(df)
        
        assert result["val"].min() >= 0
        assert result["val"].max() < 32
```

#### 6. Integration Tests

```python
class TestPipelineIntegration:
    def test_chaining_encoders(self):
        from analitica.etl import Pipeline, CSVSource
        
        pipeline = Pipeline()
        pipeline.add_source(CSVSource("sample_data.csv"))
        pipeline.add_transformer(LabelEncoder(columns=["category"]))
        pipeline.add_transformer(OneHotEncoder(columns=["city"]))
        
        result = pipeline.execute()
        
        assert isinstance(result, pd.DataFrame)
```

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | All 4 encoders extend BaseTransformer | Unit tests pass |
| 2 | fit() computes and stores mappings | Check `_mapping` attribute |
| 3 | transform() uses stored mappings | Compare results |
| 4 | fit_transform() combines both | Integration test |
| 5 | Unseen category handling works | Warning captured in test |
| 6 | Edge cases handled (empty, NaN) | Edge case tests |
| 7 | Exports work from package level | Import tests |
| 8 | Pipeline integration works | Pipeline tests |

---

## Dependencies

### Required (Already in project)

- `pandas>=2.0.0` - DataFrame operations
- `numpy>=1.24.0` - Numerical operations
- `analitica.etl.transformer.BaseTransformer` - Base class

### No New Dependencies

All required packages are already declared in `pyproject.toml`.

---

## Implementation Order

1. Create `base.py` - BaseEncoder abstract class
2. Create `encoders.py` - LabelEncoder, OneHotEncoder, TargetEncoder, HashEncoder
3. Update `__init__.py` - Export all encoders
4. Add fixtures to `conftest.py`
5. Create `tests/test_encoders.py`
6. Run tests and verify

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TargetEncoder missing target | Medium | High | Clear ValueError with column list |
| Unseen categories in transform | Medium | Low | Warning + default value |
| High-cardinality memory | Low | Medium | HashEncoder uses fixed bins |
| OneHotEncoder feature explosion | Medium | Medium | drop_first option, warn if many cats |

---

## References

- [scikit-learn Preprocessing - Encoding Categoricals](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Target Encoding - Kaggle](https://www.kaggle.com/wiki/MeanEncoding)
- [Feature Hashing](https://en.wikipedia.org/wiki/Feature_hashing)
