# Technical Design: Data Normalizers

## Overview

This document describes the technical design for implementing data normalizers in the `analitica` library. The normalizers provide standardized preprocessing transformers that scale numerical features for ML algorithms, improving SOM convergence and clustering quality.

## Architecture

### Class Hierarchy

```
BaseTransformer (from etl/transformer.py)
    │
    └── BaseNormalizer (normalization/base.py)
            │
            ├── MinMaxScaler (normalization/scalers.py)
            ├── ZScoreScaler (normalization/scalers.py)
            ├── RobustScaler (normalization/scalers.py)
            ├── LogTransformer (normalization/transformers.py)
            └── PowerTransformer (normalization/transformers.py)
```

### Design Patterns

1. **sklearn-compatible API**: All normalizers follow scikit-learn's fit/transform/fit_transform pattern
2. **Immutable fit parameters**: Statistics computed during `fit()` are stored and reused in `transform()`
3. **Column selection**: Supports both automatic numeric column detection and explicit column specification

---

## File Structure

```
src/analitica/normalization/
├── __init__.py          # Modified: Export all normalizers
├── base.py              # New: BaseNormalizer abstract class
├── scalers.py           # New: MinMaxScaler, ZScoreScaler, RobustScaler
└── transformers.py      # New: LogTransformer, PowerTransformer
```

---

## Data Flow

### Normalizer Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     fit(data)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Select columns (explicit or auto-detect numeric) │   │
│  │ 2. Compute normalization parameters                 │   │
│  │ 3. Store parameters in self._params                  │   │
│  │ 4. Return self                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   transform(data)                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Copy input DataFrame                             │   │
│  │ 2. Apply normalization using stored params          │   │
│  │ 3. Return transformed DataFrame                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Integration

Normalizers integrate with the existing `Pipeline` class:

```python
from analitica.etl import Pipeline
from analitica.normalization import MinMaxScaler, ZScoreScaler

pipeline = Pipeline()
pipeline.add_source(CSVSource("data.csv"))
pipeline.add_transformer(MinMaxScaler(columns=["age", "score"]))
pipeline.add_transformer(ZScoreScaler())
pipeline.execute()
```

---

## Implementation Details

### 1. BaseNormalizer (base.py)

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any

from analitica.etl.transformer import BaseTransformer


class BaseNormalizer(BaseTransformer):
    """Base class for all normalizers.
    
    Implements sklearn-compatible fit/transform pattern with
    automatic numeric column detection.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize the normalizer.
        
        Args:
            columns: Optional list of column names to normalize.
                    If None, all numeric columns are processed.
        """
        self.columns = columns
        self._fitted_columns: List[str] = []
        self._params: Dict[str, Any] = {}
    
    @abstractmethod
    def _compute_params(self, data: pd.DataFrame) -> dict:
        """Compute normalization parameters from data.
        
        Args:
            data: DataFrame with only the columns to normalize.
            
        Returns:
            Dictionary containing normalization parameters.
        """
        pass
    
    @abstractmethod
    def _apply_transform(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Apply the normalization transformation.
        
        Args:
            data: DataFrame with only the columns to transform.
            params: Parameters computed during fit().
            
        Returns:
            Transformed DataFrame.
        """
        pass
    
    def fit(self, data: pd.DataFrame) -> "BaseNormalizer":
        """Learn normalization parameters from data.
        
        Args:
            data: The input DataFrame.
            
        Returns:
            self: The fitted normalizer.
        """
        columns = self.columns or self._get_numeric_columns(data)
        if not columns:
            raise ValueError("No numeric columns found for normalization")
        
        self._fitted_columns = columns
        self._params = self._compute_params(data[columns])
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using learned parameters.
        
        Args:
            data: The input DataFrame.
            
        Returns:
            pd.DataFrame: The transformed data.
            
        Raises:
            TransformerError: If normalizer has not been fitted.
        """
        if not self._fitted_columns:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
        
        result = data.copy()
        result[self._fitted_columns] = self._apply_transform(
            data[self._fitted_columns], 
            self._params
        )
        return result
    
    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of numeric columns from DataFrame.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            List of numeric column names.
        """
        return data.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_params(self) -> Dict[str, Any]:
        """Get fitted parameters.
        
        Returns:
            Dictionary of fitted parameters.
        """
        return self._params.copy()
```

### 2. Scalers (scalers.py)

```python
"""Scaling normalizers for numerical data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from .base import BaseNormalizer


class MinMaxScaler(BaseNormalizer):
    """Scale features to a fixed range [0, 1].
    
    Transforms each feature to lie between min and max values.
    Preserves the original distribution shape.
    
    Formula: X_scaled = (X - X_min) / (X_max - X_min)
    """
    
    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute min and max for each column.
        
        Args:
            data: DataFrame with columns to normalize.
            
        Returns:
            Dictionary mapping column names to {min, max} values.
        """
        return {
            col: {"min": float(data[col].min()), "max": float(data[col].max())}
            for col in data.columns
        }
    
    def _apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply MinMax scaling.
        
        Args:
            data: DataFrame with columns to transform.
            params: Dictionary with min/max values per column.
            
        Returns:
            Scaled DataFrame.
        """
        result = data.copy()
        for col in data.columns:
            col_params = params[col]
            col_range = col_params["max"] - col_params["min"]
            if col_range == 0:
                # Handle constant column (no variance)
                result[col] = 0.0
            else:
                result[col] = (data[col] - col_params["min"]) / col_range
        return result


class ZScoreScaler(BaseNormalizer):
    """Standardize features to have zero mean and unit variance.
    
    Also known as StandardScaler in sklearn.
    
    Formula: X_scaled = (X - mean) / std
    """
    
    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute mean and standard deviation for each column.
        
        Args:
            data: DataFrame with columns to normalize.
            
        Returns:
            Dictionary mapping column names to {mean, std} values.
        """
        return {
            col: {
                "mean": float(data[col].mean()), 
                "std": float(data[col].std())
            }
            for col in data.columns
        }
    
    def _apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply Z-score standardization.
        
        Args:
            data: DataFrame with columns to transform.
            params: Dictionary with mean/std values per column.
            
        Returns:
            Standardized DataFrame.
        """
        result = data.copy()
        for col in data.columns:
            col_params = params[col]
            if col_params["std"] == 0:
                # Handle constant column (zero variance)
                result[col] = 0.0
            else:
                result[col] = (data[col] - col_params["mean"]) / col_params["std"]
        return result


class RobustScaler(BaseNormalizer):
    """Scale features using median and interquartile range.
    
    Outlier-resistant alternative to ZScoreScaler.
    Uses median (50th percentile) and IQR (Q3 - Q1).
    
    Formula: X_scaled = (X - median) / IQR
    """
    
    def __init__(self, columns: List[str] = None, quantile_range: tuple = (25, 75)):
        """Initialize RobustScaler.
        
        Args:
            columns: Optional list of column names to normalize.
            quantile_range: Quantile range as (lower, upper) percentiles.
        """
        super().__init__(columns)
        self.quantile_range = quantile_range
    
    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute median and IQR for each column.
        
        Args:
            data: DataFrame with columns to normalize.
            
        Returns:
            Dictionary mapping column names to {median, iqr} values.
        """
        lower, upper = self.quantile_range
        return {
            col: {
                "median": float(data[col].median()),
                "iqr": float(data[col].quantile(upper / 100) - data[col].quantile(lower / 100))
            }
            for col in data.columns
        }
    
    def _apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply robust scaling.
        
        Args:
            data: DataFrame with columns to transform.
            params: Dictionary with median/IQR values per column.
            
        Returns:
            Scaled DataFrame.
        """
        result = data.copy()
        for col in data.columns:
            col_params = params[col]
            if col_params["iqr"] == 0:
                # Handle column with zero IQR
                result[col] = 0.0
            else:
                result[col] = (data[col] - col_params["median"]) / col_params["iqr"]
        return result
```

### 3. Transformers (transformers.py)

```python
"""Transformation normalizers for non-linear data distributions."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from .base import BaseNormalizer


class LogTransformer(BaseNormalizer):
    """Apply logarithmic transformation to reduce skewness.
    
    Handles negative and zero values by shifting data before log transform.
    
    Formula: X_scaled = log(X + shift)  (natural log)
    """
    
    def __init__(self, columns: List[str] = None, base: str = "natural"):
        """Initialize LogTransformer.
        
        Args:
            columns: Optional list of column names to transform.
            base: Logarithm base - "natural", "base10", or "base2".
        """
        super().__init__(columns)
        self.base = base
        self._log_func = {
            "natural": np.log,
            "base10": np.log10,
            "base2": np.log2,
        }.get(base, np.log)
    
    def _compute_params(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute shift value for each column to handle zeros/negatives.
        
        Args:
            data: DataFrame with columns to transform.
            
        Returns:
            Dictionary mapping column names to shift values.
        """
        shifts = {}
        for col in data.columns:
            min_val = data[col].min()
            if min_val <= 0:
                # Shift to make all values positive
                shifts[col] = abs(min_val) + 1
            else:
                shifts[col] = 0
        return shifts
    
    def _apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply log transformation.
        
        Args:
            data: DataFrame with columns to transform.
            params: Dictionary with shift values per column.
            
        Returns:
            Transformed DataFrame.
        """
        result = data.copy()
        for col in data.columns:
            shift = params[col]
            if shift > 0:
                result[col] = self._log_func(data[col] + shift)
            else:
                # Use log1p for better numerical stability with small values
                result[col] = self._log_func(data[col].clip(lower=1e-10))
        return result


class PowerTransformer(BaseNormalizer):
    """Apply Yeo-Johnson power transformation to make data more Gaussian.
    
    Handles all value ranges including negative values.
    Wraps sklearn's PowerTransformer for the actual computation.
    """
    
    def __init__(self, columns: List[str] = None, method: str = "yeo-johnson"):
        """Initialize PowerTransformer.
        
        Args:
            columns: Optional list of column names to transform.
            method: Transformation method - "yeo-johnson" (only supported).
        """
        super().__init__(columns)
        self.method = method
        
        # Import here to make sklearn optional for other normalizers
        from sklearn.preprocessing import PowerTransformer as SKPowerTransformer
        self._sklearn_transformer = SKPowerTransformer(method=method, standardize=False)
    
    def _compute_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute Yeo-Johnson transformation parameters.
        
        Args:
            data: DataFrame with columns to transform.
            
        Returns:
            Dictionary with lambdas and fitted transformer.
        """
        # Fit sklearn transformer on data
        self._sklearn_transformer.fit(data)
        
        return {
            "lambdas": dict(zip(data.columns, self._sklearn_transformer.lambdas_)),
            "sklearn_transformer": self._sklearn_transformer
        }
    
    def _apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply power transformation.
        
        Args:
            data: DataFrame with columns to transform.
            params: Dictionary with transformation parameters.
            
        Returns:
            Transformed DataFrame.
        """
        sklearn_transformer = params["sklearn_transformer"]
        transformed = sklearn_transformer.transform(data)
        return pd.DataFrame(transformed, columns=data.columns, index=data.index)
```

### 4. Module Exports (__init__.py)

```python
"""Normalization Module - Técnicas de normalización de datos"""

from .base import BaseNormalizer
from .scalers import MinMaxScaler, ZScoreScaler, RobustScaler
from .transformers import LogTransformer, PowerTransformer

__all__ = [
    "BaseNormalizer",
    "MinMaxScaler", 
    "ZScoreScaler",
    "RobustScaler",
    "LogTransformer",
    "PowerTransformer",
]
```

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| No numeric columns | Raise `ValueError` with descriptive message |
| Not fitted before transform | Raise `RuntimeError` |
| Zero range (max == min) | Return 0.0 for that column |
| Zero variance (std == 0) | Return 0.0 for that column |
| Zero IQR | Return 0.0 for that column |
| Empty DataFrame | Allow through (returns empty DataFrame) |

---

## Testing Strategy

### Test File Structure

```
tests/
├── test_normalization.py    # Main test file
```

### Test Categories

#### 1. Unit Tests for BaseNormalizer

```python
# tests/test_normalization.py

import pytest
import pandas as pd
import numpy as np
from analitica.normalization import (
    BaseNormalizer, MinMaxScaler, ZScoreScaler, 
    RobustScaler, LogTransformer, PowerTransformer
)


class TestBaseNormalizer:
    """Test cases for BaseNormalizer base class."""
    
    def test_fit_stores_columns(self):
        """fit() stores the fitted column names."""
        scaler = MinMaxScaler()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        scaler.fit(df)
        
        assert scaler._fitted_columns == ["a", "b"]
    
    def test_fit_with_explicit_columns(self):
        """fit() uses explicitly specified columns."""
        scaler = MinMaxScaler(columns=["a"])
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        scaler.fit(df)
        
        assert scaler._fitted_columns == ["a"]
    
    def test_transform_raises_if_not_fitted(self):
        """transform() raises RuntimeError if not fitted."""
        scaler = MinMaxScaler()
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(RuntimeError):
            scaler.transform(df)
    
    def test_fit_transform_combines_operations(self):
        """fit_transform() works as convenience method."""
        scaler = MinMaxScaler()
        df = pd.DataFrame({"a": [0, 50, 100]})
        
        result = scaler.fit_transform(df)
        
        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0
```

#### 2. Scaler Tests

```python
class TestMinMaxScaler:
    """Test cases for MinMaxScaler."""
    
    def test_scale_to_unit_range(self):
        """Scales data to [0, 1] range."""
        scaler = MinMaxScaler()
        df = pd.DataFrame({"a": [0, 50, 100]})
        
        result = scaler.fit_transform(df)
        
        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0
        assert result["a"].iloc[1] == 0.5
    
    def test_handles_constant_column(self):
        """Handles column with zero range."""
        scaler = MinMaxScaler()
        df = pd.DataFrame({"a": [5, 5, 5]})
        
        result = scaler.fit_transform(df)
        
        assert result["a"].iloc[0] == 0.0
    
    def test_preserves_other_columns(self):
        """Non-numeric columns are preserved."""
        scaler = MinMaxScaler()
        df = pd.DataFrame({"a": [0, 100], "b": ["x", "y"]})
        
        result = scaler.fit_transform(df)
        
        assert list(result["b"]) == ["x", "y"]


class TestZScoreScaler:
    """Test cases for ZScoreScaler."""
    
    def test_standardizes_to_mean_zero(self):
        """Data is standardized to mean=0, std=1."""
        scaler = ZScoreScaler()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        
        result = scaler.fit_transform(df)
        
        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["a"].std() - 1.0) < 1e-10
    
    def test_handles_zero_variance(self):
        """Handles column with zero variance."""
        scaler = ZScoreScaler()
        df = pd.DataFrame({"a": [5, 5, 5]})
        
        result = scaler.fit_transform(df)
        
        assert result["a"].iloc[0] == 0.0


class TestRobustScaler:
    """Test cases for RobustScaler."""
    
    def test_uses_median_and_iqr(self):
        """Scales using median and IQR."""
        scaler = RobustScaler()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})  # Outlier
        
        result = scaler.fit_transform(df)
        
        assert result["a"].median() == pytest.approx(0.0, abs=0.1)
```

#### 3. Transformer Tests

```python
class TestLogTransformer:
    """Test cases for LogTransformer."""
    
    def test_handles_negative_values(self):
        """Shifts data to handle negative values."""
        transformer = LogTransformer()
        df = pd.DataFrame({"a": [-5, 0, 5]})
        
        result = transformer.fit_transform(df)
        
        assert result["a"].min() > -np.inf
    
    def test_reduces_skewness(self):
        """Reduces right-skewed data skewness."""
        transformer = LogTransformer()
        # Highly skewed data
        df = pd.DataFrame({"a": [1, 10, 100, 1000, 10000]})
        
        result = transformer.fit_transform(df)
        
        original_skew = df["a"].skew()
        transformed_skew = result["a"].skew()
        assert abs(transformed_skew) < abs(original_skew)


class TestPowerTransformer:
    """Test cases for PowerTransformer."""
    
    def test_makes_data_more_gaussian(self):
        """Yeo-Johnson makes data more Gaussian-like."""
        transformer = PowerTransformer()
        # Exponential distribution
        df = pd.DataFrame({"a": np.random.exponential(2, 1000)})
        
        result = transformer.fit_transform(df)
        
        # Check skewness is closer to 0
        original_skew = abs(df["a"].skew())
        transformed_skew = abs(result["a"].skew())
        assert transformed_skew < original_skew
```

#### 4. Integration Tests

```python
class TestPipelineIntegration:
    """Test normalizers in Pipeline context."""
    
    def test_chaining_normalizers(self):
        """Multiple normalizers can be chained."""
        from analitica.etl import Pipeline, CSVSource
        
        # This test requires a sample CSV fixture
        # See conftest.py for fixture definition
        pipeline = Pipeline()
        pipeline.add_source(CSVSource("sample_data.csv"))
        pipeline.add_transformer(MinMaxScaler(columns=["numeric_col"]))
        pipeline.add_transformer(ZScoreScaler())
        
        # Pipeline should execute without error
        result = pipeline.execute()
        
        assert isinstance(result, pd.DataFrame)
```

### Test Fixtures (conftest.py additions)

```python
@pytest.fixture
def numeric_dataframe():
    """Provide a DataFrame with numeric columns for normalization tests."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "score": [65.0, 75.5, 82.3, 91.0, 55.5],
        "height": [170, 165, 180, 175, 168],
    })


@pytest.fixture
def skewed_dataframe():
    """Provide a DataFrame with skewed distribution."""
    return pd.DataFrame({
        "income": [20000, 25000, 30000, 500000, 35000],  # Outlier
        "orders": [1, 2, 3, 4, 5, 10, 15, 20, 50],  # Right-skewed
    })
```

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | All 5 normalizers extend BaseTransformer | Unit tests pass |
| 2 | fit() computes and stores parameters | Check `get_params()` |
| 3 | transform() uses stored parameters | Compare results |
| 4 | fit_transform() combines both | Integration test |
| 5 | Column selection works | Explicit columns test |
| 6 | Edge cases handled | Zero variance, empty, etc. |
| 7 | Exports work from package level | Import tests |
| 8 | Pipeline integration works | Pipeline tests |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Division by zero | Low | Medium | Check for zero range/variance/IQR before division |
| Negative values in log | Low | Medium | Automatic shift calculation in LogTransformer |
| Empty DataFrame | Low | Low | Allow through, return empty DataFrame |
| Non-numeric columns | Medium | Low | Automatic filtering, explicit column selection |

---

## Dependencies

### Required

- `pandas>=2.0.0` - Already in dependencies
- `numpy>=1.24.0` - Already in dependencies  
- `scikit-learn>=1.3.0` - Already in dependencies (for PowerTransformer)

### No New Dependencies

All required packages are already declared in `pyproject.toml`.

---

## Implementation Order

1. Create `base.py` - BaseNormalizer abstract class
2. Create `scalers.py` - MinMaxScaler, ZScoreScaler, RobustScaler
3. Create `transformers.py` - LogTransformer, PowerTransformer
4. Update `__init__.py` - Export all normalizers
5. Add fixtures to `conftest.py`
6. Create `tests/test_normalization.py`
7. Run tests and verify

---

## References

- [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [Yeo-Johnson Transformation](https://www.statisticshowto.com/yeo-johnson-transformation/)
