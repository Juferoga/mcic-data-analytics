# Tasks: Data Normalizers

## Phase 1: Base Normalizer

- [x] 1.1 Create `src/analitica/normalization/base.py` with BaseNormalizer class extending BaseTransformer
- [x] 1.2 Update `src/analitica/normalization/__init__.py` to export BaseNormalizer

## Phase 2: Scalers

- [x] 2.1 Create `src/analitica/normalization/scalers.py` with MinMaxScaler
- [x] 2.2 Add ZScoreScaler to scalers.py
- [x] 2.3 Add RobustScaler to scalers.py
- [x] 2.4 Update `src/analitica/normalization/__init__.py` exports

## Phase 3: Transformers

- [x] 3.1 Create `src/analitica/normalization/transformers.py` with LogTransformer
- [x] 3.2 Add PowerTransformer to transformers.py
- [x] 3.3 Update `src/analitica/normalization/__init__.py` exports

## Phase 4: Tests

- [x] 4.1 Create `tests/test_normalization.py` with tests for BaseNormalizer
- [x] 4.2 Add tests for MinMaxScaler
- [x] 4.3 Add tests for ZScoreScaler
- [x] 4.4 Add tests for RobustScaler
- [x] 4.5 Add tests for LogTransformer
- [x] 4.6 Add tests for PowerTransformer
- [x] 4.7 Run pytest to verify all tests pass

## Phase 5: Demo

- [x] 5.1 Create `examples/demo_normalizers.py` showing all normalizers in action
- [ ] 5.2 Update README.md with normalizer examples
