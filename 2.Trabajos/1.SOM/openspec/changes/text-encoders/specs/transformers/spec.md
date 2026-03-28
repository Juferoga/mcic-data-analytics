# Text Encoders Specification

## Purpose

Define categorical/text encoding transformers that convert categorical data to numerical representations for ML algorithms. Essential for SOM and clustering algorithms that require numerical input.

---

## ADDED Requirements

### Requirement: BaseEncoder Interface

All encoders **MUST** extend `BaseTransformer` from `src/analitica/transformers/base.py`.

The `BaseEncoder` abstract class **SHALL** provide:

- `fit(data: pd.DataFrame, columns: List[str] | None = None)`: Learn encoding mapping from data
- `transform(data: pd.DataFrame) -> pd.DataFrame`: Apply learned encoding to data
- `fit_transform(data: pd.DataFrame, columns: List[str] | None = None) -> pd.DataFrame`: Convenience method

The `columns` parameter **MUST**:
- Accept `None` to encode all object/category columns
- Accept a list of column names to encode specific columns only

The encoder **MUST** handle non-categorical columns gracefully by preserving them unchanged in the output.

#### Scenario: BaseEncoder with all columns

- GIVEN a DataFrame with categorical columns `[city, status]` and numeric column `value`
- WHEN `fit(data, columns=None)` is called
- THEN the encoder learns parameters from categorical columns `city, status` only
- AND column `value` is preserved unchanged through transform

#### Scenario: BaseEncoder with specific columns

- GIVEN a DataFrame with columns `[city, status, value]`
- WHEN `fit(data, columns=['city'])` is called
- THEN the encoder learns parameters for column `city` only
- AND column `status` remains unmodified during transform

#### Scenario: BaseEncoder fit on empty data

- GIVEN an encoder and an empty DataFrame
- WHEN `fit(data)` is called
- THEN the system **MUST** raise `EncoderError` with descriptive message

#### Scenario: BaseEncoder transform before fit

- GIVEN an encoder that has NOT been fitted
- WHEN `transform(data)` is called
- THEN the system **MUST** raise `EncoderError` with message indicating the encoder was not fitted

---

### Requirement: LabelEncoder

The `LabelEncoder` **SHALL** perform ordinal encoding, converting categorical labels to integer values while preserving order for ordinal data.

The encoder **MUST**:
- Assign integer labels starting from 0 in sorted order of unique categories
- Preserve the order of first appearance OR use sorted order for ordinal data
- Store the mapping from category to integer for transform phase

#### Scenario: LabelEncoder basic encoding

- GIVEN a LabelEncoder fitted on data with categories `['low', 'medium', 'high']`
- WHEN `transform(data)` is called with category `'medium'`
- THEN the result **MUST** be `1`

- GIVEN a LabelEncoder fitted on data with categories `['low', 'medium', 'high']`
- WHEN `transform(data)` is called with category `'high'`
- THEN the result **MUST** be `2`

#### Scenario: LabelEncoder preserves order

- GIVEN a LabelEncoder fitted on ordinal data with categories `['junior', 'mid', 'senior']`
- WHEN categories are encoded
- THEN junior < mid < senior in numeric representation

#### Scenario: LabelEncoder handles unseen categories

- GIVEN a LabelEncoder fitted on categories `['a', 'b', 'c']`
- WHEN `transform(data)` is called with unseen category `'d'`
- THEN the system **MUST** raise `EncoderError` by default
- OR if configured with `handle_unknown='use_default_value'`, use fallback value

#### Scenario: LabelEncoder preserves non-categorical columns

- GIVEN a LabelEncoder fitted on categorical column `priority`
- WHEN `transform(data)` is called on DataFrame with additional column `score`
- THEN column `score` values **MUST** remain unchanged

#### Scenario: LabelEncoder fit_transform

- GIVEN a LabelEncoder
- WHEN `fit_transform(data)` is called
- THEN it learns the category mapping and returns encoded data in one step

---

### Requirement: OneHotEncoder

The `OneHotEncoder` **SHALL** perform nominal encoding, creating binary columns for each unique category.

The encoder **MUST**:
- Create one binary column per unique category
- Use column naming convention: `{original_column}_{category}` 
- Drop the original column from output
- Store unique categories learned during fit

#### Scenario: OneHotEncoder creates binary columns

- GIVEN a OneHotEncoder fitted on data with categories `['red', 'green', 'blue']`
- WHEN `transform(data)` is called with category `'red'`
- THEN the output **MUST** contain columns `color_red`, `color_green`, `color_blue`
- AND `color_red` = 1, `color_green` = 0, `color_blue` = 0

#### Scenario: OneHotEncoder handles multiple categories in row

- GIVEN a OneHotEncoder fitted on column `city` with categories `['NYC', 'LA', 'SF']`
- WHEN `transform(data)` is called
- THEN the original `city` column **MUST** be dropped from output
- AND only binary columns remain

#### Scenario: OneHotEncoder handles unseen categories

- GIVEN a OneHotEncoder fitted on categories `['a', 'b', 'c']`
- WHEN `transform(data)` is called with unseen category `'d'`
- THEN if `handle_unknown='ignore'`: set all output columns to 0
- OR if `handle_unknown='error'` (default): raise `EncoderError`

#### Scenario: OneHotEncoder handles sparse output

- GIVEN a OneHotEncoder configured with `sparse=True`
- WHEN `transform(data)` is called
- THEN the output **SHOULD** be a sparse matrix

#### Scenario: OneHotEncoder preserves non-categorical columns

- GIVEN a OneHotEncoder fitted on categorical column `city`
- WHEN `transform(data)` is called on DataFrame with additional column `population`
- THEN column `population` values **MUST** remain unchanged

---

### Requirement: TargetEncoder

The `TargetEncoder` **SHALL** perform supervised encoding, replacing categories with the mean of the target variable.

The encoder **MUST**:
- Calculate mean target value per category during fit
- Apply smoothing to prevent overfitting on rare categories
- Store category means and global mean for transform phase

The smoothing formula **SHALL** be:
```
smoothed_mean = (category_count * category_mean + smoothing * global_mean) / (category_count + smoothing)
```

#### Scenario: TargetEncoder encodes with target mean

- GIVEN a TargetEncoder fitted on categories with target values:
  - category 'a': mean = 10.0 (count = 100)
  - category 'b': mean = 20.0 (count = 100)
  - global mean = 15.0
- WHEN `transform(data)` is called with category `'a'`
- THEN the result **MUST** be approximately `10.0`

#### Scenario: TargetEncoder applies smoothing

- GIVEN a TargetEncoder fitted with `smoothing=10` on categories:
  - category 'rare': mean = 100.0 (count = 1)
  - global mean = 50.0
- WHEN `transform(data)` is called with category `'rare'`
- THEN the result **MUST** be closer to global mean than 100.0 due to smoothing

#### Scenario: TargetEncoder handles unknown categories

- GIVEN a TargetEncoder fitted on categories `['a', 'b', 'c']` with global mean = 5.0
- WHEN `transform(data)` is called with unseen category `'d'`
- THEN the result **MUST** be the global mean `5.0`

#### Scenario: TargetEncoder requires target variable

- GIVEN a TargetEncoder
- WHEN `fit(data)` is called without target variable `y`
- THEN the system **MUST** raise `EncoderError` with message indicating target is required

#### Scenario: TargetEncoder handles missing target values

- GIVEN a TargetEncoder and data with NaN in target column
- WHEN `fit(data, y)` is called
- THEN the encoder **MUST** exclude rows with NaN target from mean calculation
- AND raise warning about excluded rows

---

### Requirement: HashEncoder

The `HashEncoder` **SHALL** perform hashing encoding for high-cardinality features, mapping categories to a fixed number of buckets.

The encoder **MUST**:
- Use a consistent hash function to map categories to bucket indices
- Create multiple hash columns (n_hash_functions) for reduced collision
- Store the number of buckets for transform phase

The output **SHALL** be:
- `n_hash_functions` columns named `{column}_hash_{i}` where i = 0 to n-1
- Each column contains 0 or 1 indicating whether the category hashes to that bucket

#### Scenario: HashEncoder maps to buckets

- GIVEN a HashEncoder fitted with `n_buckets=100` and `n_hash_functions=3`
- WHEN `transform(data)` is called
- THEN the output **MUST** contain 3 hash columns per original column

#### Scenario: HashEncoder handles collisions

- GIVEN a HashEncoder fitted with `n_buckets=10`
- WHEN two different categories both hash to bucket 3
- THEN both categories will have hash column 3 set to 1
- AND the encoder handles collisions naturally through multiple hash functions

#### Scenario: HashEncoder handles unseen categories

- GIVEN a HashEncoder fitted on training data
- WHEN `transform(data)` is called with unseen categories
- THEN the hash function produces consistent output for any category
- AND no error is raised (unlike other encoders)

#### Scenario: HashEncoder handles high cardinality

- GIVEN a HashEncoder with `n_buckets=1000` fitted on data with 10,000 unique categories
- WHEN `transform(data)` is called
- THEN the output dimension remains fixed at `n_hash_functions` columns
- AND no memory issues occur due to high cardinality

#### Scenario: HashEncoder with custom hash function

- GIVEN a HashEncoder configured with custom hash function
- WHEN categories are encoded
- THEN the custom hash function **MUST** be used instead of default hash

---

### Requirement: Encoder Error Handling

The system **MUST** define an `EncoderError` exception class for encoding-specific errors.

#### Scenario: EncoderError raised on invalid operation

- GIVEN a LabelEncoder that has NOT been fitted
- WHEN `transform(data)` is called
- THEN the system **MUST** raise `EncoderError` with message indicating the encoder was not fitted

---

## Data Flow

```
Input DataFrame → [fit() learns encoding mapping] → [transform() applies encoding] → Output DataFrame
```

All encoders **SHALL**:
- Accept pandas DataFrame as input
- Return pandas DataFrame as output
- Preserve column order and non-categorical columns
- Support fit_transform convenience method

---

## Error Handling

| Error Type | Condition | Behavior |
|------------|-----------|----------|
| `EncoderError` | fit() called with empty data | Raised with descriptive message |
| `EncoderError` | transform() called before fit() | Raised with descriptive message |
| `EncoderError` | Invalid columns specified | Raised with descriptive message |
| `EncoderError` | LabelEncoder sees unseen category (default) | Raised with message about unknown category |
| `EncoderError` | OneHotEncoder sees unseen category (default) | Raised with message about unknown category |
| `EncoderError` | TargetEncoder fit() without y parameter | Raised with message indicating target is required |

---

## Success Criteria

- [ ] BaseEncoder extends BaseTransformer with fit/transform/fit_transform methods
- [ ] columns parameter supports None (all) and list (specific) modes
- [ ] Non-categorical columns are preserved unchanged
- [ ] LabelEncoder assigns integer labels in sorted order
- [ ] LabelEncoder preserves ordinal order
- [ ] LabelEncoder raises error or uses fallback for unseen categories
- [ ] OneHotEncoder creates binary columns with proper naming
- [ ] OneHotEncoder drops original column from output
- [ ] OneHotEncoder handles unseen categories (ignore or error)
- [ ] TargetEncoder encodes with mean of target per category
- [ ] TargetEncoder applies smoothing for rare categories
- [ ] TargetEncoder uses global mean for unknown categories
- [ ] TargetEncoder requires target variable y
- [ ] HashEncoder maps to fixed number of buckets
- [ ] HashEncoder produces multiple hash columns
- [ ] HashEncoder handles any category (no unseen errors)
- [ ] All encoders raise EncoderError on appropriate errors
