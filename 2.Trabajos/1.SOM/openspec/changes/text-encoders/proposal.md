# Proposal: Text-to-Number Encoders

## Intent

Implement transformers that convert categorical/text data to numerical representations for ML algorithms. Essential for SOM and clustering algorithms that require numerical input. Addresses the gap in categorical data encoding capabilities needed by downstream ML pipelines.

## Scope

### In Scope
- **LabelEncoder**: Ordinal categories (low/medium/high → 0/1/2)
- **OneHotEncoder**: Nominal categories (city → city_A, city_B, city_C columns)
- **TargetEncoder**: Encode with target variable mean (supervised)
- **HashEncoder**: For high-cardinality features (URLs, IDs)
- All encoders extend BaseTransformer interface
- Handle unseen categories in transform phase

### Out of Scope
- Binary encoders (deferred to future iteration)
- Feature union/composition utilities
- Integration with specific ML algorithms (SOM, clustering)

## Approach

Each encoder extends `BaseTransformer` from `src/analitica/etl/transformer.py`:
1. `fit()` learns the encoding mapping from training data
2. `transform()` applies the encoding to produce numerical output
3. Support column-wise encoding (single column input)
4. Handle unseen categories gracefully (encode as special value or raise)

Architecture: Follow existing normalizer pattern from `src/analitica/normalization/base.py`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/analitica/encoding/__init__.py` | New | Package exports |
| `src/analitica/encoding/base.py` | New | BaseEncoder abstract class |
| `src/analitica/encoding/label_encoder.py` | New | LabelEncoder implementation |
| `src/analitica/encoding/onehot_encoder.py` | New | OneHotEncoder implementation |
| `src/analitica/encoding/target_encoder.py` | New | TargetEncoder implementation |
| `src/analitica/encoding/hash_encoder.py` | New | HashEncoder implementation |
| `tests/test_encoder.py` | New | Unit tests for all encoders |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|-------------|
| TargetEncoder requires target variable | Medium | Document requirement, raise clear error if missing |
| High-cardinality memory issues | Low | HashEncoder uses fixed output dimension |
| Unseen categories in transform | Medium | Log warning, encode to default value |

## Rollback Plan

1. Delete `src/analitica/encoding/` directory
2. Remove exports from `src/analitica/__init__.py`
3. No database migrations or data cleanup needed (pure transformation)

## Dependencies

- pandas, numpy (already in project)
- BaseTransformer from `analitica.etl.transformer`

## Success Criteria

- [ ] All 4 encoders implemented and extend BaseTransformer
- [ ] Unit tests pass for fit/transform on each encoder
- [ ] Unseen category handling works (no exceptions)
- [ ] Integration with Pipeline works correctly
