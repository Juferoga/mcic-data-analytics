# Proposal: Data Normalizers

## Intent

Implement normalization transformers that scale numerical data for ML algorithms. These are essential preprocessing steps that bring features to comparable scales, improving SOM convergence and clustering quality. The current codebase lacks standardized normalization utilities, forcing manual preprocessing.

## Scope

### In Scope
- **MinMaxScaler**: Scale features to [0, 1] range
- **ZScoreScaler**: Standard score (mean=0, std=1)
- **RobustScaler**: Uses median and IQR (outlier-resistant)
- **LogTransformer**: Log transform for skewed data
- **PowerTransformer**: Yeo-Johnson for normalization
- All normalizers implement BaseTransformer interface
- Support column-wise and full-data normalization modes

### Out of Scope
- Online/incremental normalization (future enhancement)
- Custom normalizer creation API (deferred)
- Integration with SOM training pipeline (separate change)

## Approach

1. Create `src/analitica/normalization/base.py` with `BaseNormalizer` abstract class extending `BaseTransformer`
2. Implement each normalizer class with:
   - `fit(data)`: Compute statistics from training data
   - `transform(data)`: Apply scaling
   - `fit_transform(data)`: Convenience method
3. Use sklearn-compatible patterns for familiarity
4. Support both per-column normalization and full-data normalization via parameter

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/analitica/normalization/base.py` | New | BaseNormalizer abstract class |
| `src/analitica/normalization/scalers.py` | New | MinMaxScaler, ZScoreScaler, RobustScaler |
| `src/analitica/normalization/transformers.py` | New | LogTransformer, PowerTransformer |
| `src/analitica/normalization/__init__.py` | Modified | Export all normalizers |
| `tests/test_normalization.py` | New | Unit tests for all normalizers |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Division by zero in scaling | Low | Add epsilon/fallback in transform |
| Negative values with log | Low | Handle via abs() or shift |
| Empty data during fit | Low | Validate and raise NormalizerError |

## Rollback Plan

1. Revert changes to `src/analitica/normalization/`
2. Remove `tests/test_normalization.py`
3. Restore `__init__.py` to previous state

## Dependencies

- pandas, numpy (existing dependencies)
- sklearn (for PowerTransformer's Yeo-Johnson)

## Success Criteria

- [ ] All 5 normalizers implemented and extend BaseTransformer
- [ ] fit/transform/fit_transform methods work correctly
- [ ] Column-wise and full-data modes functional
- [ ] Unit tests pass for all normalizers
- [ ] Integration with Pipeline tested
