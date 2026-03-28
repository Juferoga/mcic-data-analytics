# Proposal: etl-engine-foundation

## Intent

Build the core ETL pipeline infrastructure for the analitica project, enabling data extraction from multiple sources, transformations via composable transformers, and loading to various destinations. This establishes the foundation for future features including SOM clustering and text-to-number conversions.

## Scope

### In Scope
- Core `Pipeline` class with fluent interface (`.extract_from().transform().load_to().run()`)
- Base `Source` abstraction with CSV implementation
- Base `Destination` abstraction with CSV implementation  
- Base `Transformer` interface with `fit_transform` pattern
- Basic CLI structure for pipeline execution
- pandas DataFrame as primary data carrier

### Out of Scope
- Database sources/destinations (CSV only)
- SOM clustering implementation
- Text-to-number transformations
- Advanced CLI options (flags, config files)
- Error handling beyond basic validation

## Approach

**Fluent Pipeline Pattern**: Build a `Pipeline` class that chains method calls:
```
pipeline = Pipeline()
pipeline.extract_from(CSVDatasource('input.csv'))
        .transform(Normalizer())
        .transform(SOMClusterer())
        .load_to(CSVDestination('output.csv'))
pipeline.run()
```

**Interface Segregation**: Define abstract base classes for extensibility:
- `Source` (interface) → `CSVDatasource` (implementation)
- `Destination` (interface) → `CSVDestination` (implementation)
- `Transformer` (interface) → concrete transformers implement `fit_transform()`

**Data Flow**: Source → [Transformer×N] → Destination, with pandas DataFrames as the data carrier between stages.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/analitica/pipeline.py` | New | Core Pipeline class with fluent interface |
| `src/analitica/sources/` | New | Base Source + CSVDatasource |
| `src/analitica/destinations/` | New | Base Destination + CSVDestination |
| `src/analitica/transformers/` | New | Base Transformer interface |
| `src/analitica/cli.py` | New | Basic CLI entry point |
| `src/analitica/__init__.py` | Modified | Export public API |
| `tests/` | New | Unit tests for core components |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| DataFrame memory issues with large files | Low | Document streaming options for future |
| Interface changes break early adopters | Medium | Keep base classes minimal, add methods via composition |
| CLI structure misaligned with future needs | Low | Use subcommand pattern for extensibility |

## Rollback Plan

1. Delete the new packages: `src/analitica/pipeline.py`, `src/analitica/sources/`, `src/analitica/destinations/`, `src/analitica/transformers/`
2. Revert `src/analitica/__init__.py` to previous state
3. Remove `src/analitica/cli.py` if created
4. Delete test files in `tests/`

All changes are additive; rollback is straightforward deletion.

## Dependencies

- Python 3.x (standard library only for v1)
- pandas (already in project requirements)

## Success Criteria

- [ ] `Pipeline().extract_from().transform().load_to().run()` executes without errors
- [ ] CSV source can read a simple CSV file into a DataFrame
- [ ] CSV destination can write a DataFrame to a CSV file
- [ ] Custom transformer can be chained into the pipeline
- [ ] CLI can invoke pipeline with source and destination paths
- [ ] Unit tests pass for core Pipeline functionality
