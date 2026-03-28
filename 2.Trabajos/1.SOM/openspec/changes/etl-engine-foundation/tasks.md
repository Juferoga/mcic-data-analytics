# Tasks: ETL Engine Foundation

## Phase 1: Core Infrastructure

- [x] 1.1 Create `src/analitica/etl/exceptions.py` with `PipelineError` base class and subclasses: `SourceError`, `DestinationError`, `TransformerError` (all inherit from `Exception`)
- [x] 1.2 Create `src/analitica/etl/source.py` with abstract `Source` class (ABC) requiring `extract() -> pd.DataFrame` and `CSVSource` implementation accepting `path: str | Path` and `**kwargs`
- [x] 1.3 Create `src/analitica/etl/destination.py` with abstract `Destination` class (ABC) requiring `save(df: pd.DataFrame)` and `CSVDestination` implementation with auto `mkdir(parents=True)`
- [x] 1.4 Create `src/analitica/etl/transformer.py` with abstract `BaseTransformer` class (ABC) requiring `fit_transform(df: pd.DataFrame) -> pd.DataFrame` and `IdentityTransformer` implementation (passthrough)

## Phase 2: Pipeline Core

- [x] 2.1 Create `src/analitica/etl/pipeline.py` with `Pipeline` class: `__init__`, `extract_from(source) -> self`, `transform(transformer) -> self`, `load_to(destination) -> self`, `run() -> self`; track `_df: Optional[pd.DataFrame]` state
- [x] 2.2 Update `src/analitica/etl/__init__.py` to export: `Pipeline`, `Source`, `Destination`, `CSVSource`, `CSVDestination`, `BaseTransformer`, `IdentityTransformer`, and exception classes

## Phase 3: Module Integration

- [x] 3.1 Update `src/analitica/__init__.py` to export `Pipeline` from `analitica.main`
- [x] 3.2 Update `src/analitica/main.py` to have an example function showing fluent pipeline usage
- [x] 3.3 Update `src/analitica/cli.py` with basic command structure for `etl run <source> <destination>`

## Phase 4: Testing

- [x] 4.1 Create `tests/__init__.py`
- [x] 4.2 Create `tests/conftest.py` with fixtures: `sample_csv`, `temp_output`, `empty_pipeline`
- [x] 4.3 Create `tests/test_pipeline.py` with tests: fluent interface chaining, successful run, error propagation
- [x] 4.4 Create `tests/test_source.py` with tests: CSVSource reads valid file, raises SourceError on invalid path
- [x] 4.5 Create `tests/test_destination.py` with tests: CSVDestination writes CSV, creates parent dirs, overwrites existing
- [x] 4.6 Create `tests/test_transformer.py` with tests: IdentityTransformer passthrough, BaseTransformer interface
- [x] 4.7 Run `pytest` to verify all tests pass

## Phase 5: Sample Data

- [x] 5.1 Create `data/samples/students.csv` with mixed numeric/categorical columns: `id`, `name`, `age`, `grade`, `major` (10-15 rows)

## Implementation Order

1. **Phase 1 first**: exceptions, source, destination, transformer — no dependencies
2. **Phase 2 next**: Pipeline depends on all Phase 1 components
3. **Phase 3 next**: Integration wiring depends on Pipeline existing
4. **Phase 4 last**: Tests verify all components work together
5. **Phase 5 final**: Sample data for integration testing and demos

## Verification

- [x] All tests pass: `pytest tests/ -v`
- [x] Fluent chain executes: `Pipeline().extract_from(src).transform(t).load_to(dst).run()`
- [x] Error scenarios raise correct exception types
