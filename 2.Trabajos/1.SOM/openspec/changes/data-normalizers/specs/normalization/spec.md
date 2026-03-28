# Data Normalizers Specification

## Purpose

Define normalization transformers that scale numerical data for ML algorithms, enabling features to be brought to comparable scales for improved SOM convergence and clustering quality.

---

## ADDED Requirements

### Requirement: BaseNormalizer Interface

All normalizers **MUST** extend `BaseTransformer` from `src/analitica/etl/transformer.py`.

The `BaseNormalizer` abstract class **SHALL** provide:

- `fit(data: pd.DataFrame, columns: List[str] | None = None)`: Learn scaling parameters from data
- `transform(data: pd.DataFrame) -> pd.DataFrame`: Apply learned scaling to data
- `fit_transform(data: pd.DataFrame, columns: List[str] | None = None) -> pd.DataFrame`: Convenience method

The `columns` parameter **MUST**:
- Accept `None` to normalize all numeric columns
- Accept a list of column names to normalize specific columns only

The normalizer **MUST** handle non-numeric columns gracefully by preserving them unchanged in the output.

#### Scenario: BaseNormalizer with all columns

- GIVEN a DataFrame with numeric columns `[a, b, c]` and non-numeric column `name`
- WHEN `fit(data, columns=None)` is called
- THEN the normalizer learns parameters from numeric columns `a, b, c` only
- AND column `name` is preserved unchanged through transform

#### Scenario: BaseNormalizer with specific columns

- GIVEN a DataFrame with columns `[a, b, c]`
- WHEN `fit(data, columns=['a', 'c'])` is called
- THEN the normalizer learns parameters for columns `a` and `c` only
- AND column `b` remains unmodified during transform

#### Scenario: BaseNormalizer fit on empty data

- GIVEN a normalizer and an empty DataFrame
- WHEN `fit(data)` is called
- THEN the system **MUST** raise `NormalizerError` with descriptive message

---

### Requirement: MinMaxScaler

The `MinMaxScaler` **SHALL** scale values to the `[0, 1]` range using min-max normalization.

The formula **SHALL** be: `transformed = (x - min) / (max - min)`

The `fit` method **MUST** store `min` and `max` values per column.

#### Scenario: MinMaxScaler normalizes data to [0, 1]

- GIVEN a MinMaxScaler fitted on data with column `value` ranging from 10 to 100
- WHEN `transform(data)` is called with `value=10`
- THEN the result **MUST** be `0.0`

- GIVEN a MinMaxScaler fitted on data with column `value` ranging from 10 to 100
- WHEN `transform(data)` is called with `value=100`
- THEN the result **MUST** be `1.0`

#### Scenario: MinMaxScaler handles constant column

- GIVEN a MinMaxScaler fitted on data where column `value` has constant value `5` (min=max=5)
- WHEN `transform(data)` is called
- THEN all values in column `value` **MUST** be set to `0.0`
- AND no division by zero error occurs

#### Scenario: MinMaxScaler preserves non-numeric columns

- GIVEN a MinMaxScaler fitted on numeric column `value`
- WHEN `transform(data)` is called on DataFrame with additional column `name`
- THEN column `name` values **MUST** remain unchanged

---

### Requirement: ZScoreScaler

The `ZScoreScaler` **SHALL** standardize features to mean=0 and standard deviation=1.

The formula **SHALL** be: `transformed = (x - mean) / std`

The `fit` method **MUST** store `mean` and `std` values per column.

#### Scenario: ZScoreScaler standardizes data

- GIVEN a ZScoreScaler fitted on data with column `value` having mean=50 and std=10
- WHEN `transform(data)` is called with `value=60`
- THEN the result **MUST** be `1.0`

- GIVEN a ZScoreScaler fitted on data with column `value` having mean=50 and std=10
- WHEN `transform(data)` is called with `value=40`
- THEN the result **MUST** be `-1.0`

#### Scenario: ZScoreScaler handles constant column

- GIVEN a ZScoreScaler fitted on data where column `value` has constant value `5` (std=0)
- WHEN `transform(data)` is called
- THEN all values in column `value` **MUST** be set to `0.0`
- AND no division by zero error occurs

---

### Requirement: RobustScaler

The `RobustScaler` **SHALL** scale features using median and interquartile range (IQR), providing outlier-resistant normalization.

The formula **SHALL** be: `transformed = (x - median) / IQR`

The `fit` method **MUST** store `median` and `IQR` (Q3 - Q1) values per column.

#### Scenario: RobustScaler normalizes with IQR

- GIVEN a RobustScaler fitted on data with column `value` having median=50 and IQR=40 (Q1=30, Q3=70)
- WHEN `transform(data)` is called with `value=90`
- THEN the result **MUST** be `1.0`

- GIVEN a RobustScaler fitted on data with column `value` having median=50 and IQR=40
- WHEN `transform(data)` is called with `value=50`
- THEN the result **MUST** be `0.0`

#### Scenario: RobustScaler handles zero IQR

- GIVEN a RobustScaler fitted on data where column `value` has IQR=0 (all values equal or insufficient variation)
- WHEN `transform(data)` is called
- THEN all values in column `value` **MUST** be set to `0.0`
- AND no division by zero error occurs

---

### Requirement: LogTransformer

The `LogTransformer` **SHALL** apply logarithmic transformation to handle skewed data distributions.

The `fit` method **MUST** detect if a shift is needed:
- If `min(data) > 0`: shift = 0
- If `min(data) <= 0`: shift = `|min(data)| + 1`

The transform **SHALL** apply: `transformed = np.log(x + shift)`

The transformer **MUST** handle negative values by using the detected shift.

#### Scenario: LogTransformer with positive values

- GIVEN a LogTransformer fitted on data where column `value` has minimum `1`
- WHEN `transform(data)` is called with `value=1`
- THEN the result **MUST** be approximately `0.0` (log(1) = 0)

- GIVEN a LogTransformer fitted on data where column `value` has minimum `1`
- WHEN `transform(data)` is called with `value=2.718`
- THEN the result **MUST** be approximately `1.0` (ln(2.718) ≈ 1)

#### Scenario: LogTransformer with zero or negative values

- GIVEN a LogTransformer fitted on data where column `value` has minimum `-5`
- WHEN `fit` is called
- THEN shift **MUST** be set to `6` (|-5| + 1)

- GIVEN a LogTransformer with shift=6 fitted on data
- WHEN `transform(data)` is called with original value `-5`
- THEN the result **MUST** be approximately `0.0` (log(-5 + 6) = log(1) = 0)

---

### Requirement: PowerTransformer

The `PowerTransformer` **SHALL** apply Yeo-Johnson transformation for normalizing skewed distributions.

The transformer **SHALL** use `sklearn.preprocessing.PowerTransformer` with `method='yeo-johnson'`.

The `fit` method **MUST** learn the optimal lambda parameter per feature.

The `transform` method **SHALL** apply the learned Yeo-Johnson transformation.

#### Scenario: PowerTransformer normalizes skewed data

- GIVEN a PowerTransformer fitted on data with heavily skewed column `value`
- WHEN `transform(data)` is called
- THEN the output column `value` **SHOULD** approximate a normal distribution

- GIVEN a PowerTransformer that was fitted on training data
- WHEN `transform(new_data)` is called with new data
- THEN it uses the lambda values learned during fit (not recalculated)

#### Scenario: PowerTransformer handles fit_transform

- GIVEN a PowerTransformer
- WHEN `fit_transform(data)` is called
- THEN it learns lambda parameters and returns transformed data in one step
- AND subsequent calls to `transform()` use the same learned parameters

---

### Requirement: Normalizer Error Handling

The system **MUST** define a `NormalizerError` exception class for normalization-specific errors.

#### Scenario: NormalizerError raised on invalid operation

- GIVEN a MinMaxScaler that has NOT been fitted
- WHEN `transform(data)` is called
- THEN the system **MUST** raise `NormalizerError` with message indicating the normalizer was not fitted

---

## Data Flow

```
Input DataFrame → [fit() learns parameters] → [transform() applies scaling] → Output DataFrame
```

All normalizers **SHALL**:
- Accept pandas DataFrame as input
- Return pandas DataFrame as output
- Preserve column order and non-numeric columns

---

## Error Handling

| Error Type | Condition | Behavior |
|------------|-----------|----------|
| `NormalizerError` | fit() called with empty data | Raised with descriptive message |
| `NormalizerError` | transform() called before fit() | Raised with descriptive message |
| `NormalizerError` | Invalid columns specified | Raised with descriptive message |

---

## Success Criteria

- [ ] BaseNormalizer extends BaseTransformer with fit/transform/fit_transform methods
- [ ] columns parameter supports None (all) and list (specific) modes
- [ ] Non-numeric columns are preserved unchanged
- [ ] MinMaxScaler scales to [0, 1] range correctly
- [ ] MinMaxScaler handles constant columns (sets to 0)
- [ ] ZScoreScaler produces mean=0, std=1
- [ ] ZScoreScaler handles constant columns (sets to 0)
- [ ] RobustScaler uses median and IQR
- [ ] RobustScaler handles zero IQR (sets to 0)
- [ ] LogTransformer handles positive values
- [ ] LogTransformer handles zero and negative values with shift
- [ ] PowerTransformer uses Yeo-Johnson from sklearn
- [ ] All normalizers raise NormalizerError on appropriate errors
